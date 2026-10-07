//! Checker-owned Python payload serialization for native Racecheck.

use std::future::Future;
use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::race_shadow::{
    PhysicalRaceOrderingDomain, PhysicalRaceOrderingFailure, PhysicalRaceProxyDomain,
};
use crate::runtime::{ExecutionPolicy, LaunchSelection};
use crate::sync_check_python::{
    analysis_scope_to_py, dynamic_op_to_py, engine_error_is_incomplete, engine_error_to_py,
    engine_incomplete_to_py, execution_stats_to_py, has_cluster_barrier_warp_exit_incomplete,
    physical_barrier_to_py, reported_sync_execution_error, set_optional_usize,
    setmaxnreg_deadlock_is_partial_domain_incomplete, sync_incomplete_to_py, sync_result_to_py,
    topology_to_py,
};
use crate::{
    AliasStaleReadAdvisory, PhysicalAllocationId, AnalysisGapDomain, EngineError, ExecutionReport,
    DeclaredWordBypassDiagnostic, GlobalScopeMismatchDiagnostic, LanePhysicalAccess,
    UndeclaredProtocolWordDiagnostic,
    PhysicalByteSpan, PhysicalMemory,
    PhysicalRaceFinding, PhysicalRaceKind, PhysicalRaceWitness, RaceCheckAccessRecord,
    RaceCheckIncompleteReason, RaceCheckLaunchState, RaceCheckMode, RaceCheckResult,
    RaceCheckStatus, SyncCheckIncompleteReason, SyncCheckStatus, WarpEngine,
};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

const SAME_THREAD_REGISTER_DEPENDENCY_PTX_URL: &str = "https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-memory-consistency-model-canonical-sync-patterns-reg-dependency-same-thread";

/// Verify a source-derived seed, rebuilding invocation memory before a retry.
/// The closure's true arm must include every bound global allocation.
pub fn with_complete_global_write_seed(
    py: Python<'_>,
    mut run: impl FnMut(bool) -> PyResult<Py<PyAny>>,
) -> PyResult<Py<PyAny>> {
    let first = run(false)?;
    let mut missed = false;
    for reason in first.bind(py).get_item("incomplete")?.try_iter()? {
        if reason?.get_item("reason")?.extract::<String>()? == "global_write_seed_incomplete" {
            missed = true;
            break;
        }
    }
    if !missed {
        return Ok(first);
    }
    // Exactly one retry: an incomplete full seed remains an incomplete result.
    let result = run(true)?;
    let timing = result.bind(py).get_item("timing")?.cast_into::<PyDict>()?;
    for (key, value) in first.bind(py).get_item("timing")?.cast_into::<PyDict>()?.iter() {
        let total = value.extract::<u64>()?.saturating_add(
            timing.get_item(&key)?.expect("same timing schema").extract::<u64>()?,
        );
        timing.set_item(key, total)?;
    }
    result.bind(py).get_item("stats")?.set_item("global_write_seed_replays", 1_u32)?;
    Ok(result)
}

/// Execute and serialize one direct online-vector-clock Racecheck phase.
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_native_race_check_phase<MakeWarp, WarpFuture>(
    py: Python<'_>,
    phase_index: usize,
    phase_name: &str,
    topology: crate::LaunchTopology,
    physical: PhysicalMemory,
    global_write_allocations: Vec<PhysicalAllocationId>,
    selection: LaunchSelection,
    max_workers: usize,
    execution_policy: ExecutionPolicy,
    inspect_accesses: bool,
    global_memory_model_enabled: bool,
    max_polls: Option<usize>,
    max_transitions: Option<usize>,
    mut make_warp: MakeWarp,
) -> PyResult<Py<PyAny>>
where
    MakeWarp: FnMut(WarpEngine<RaceCheckMode>) -> WarpFuture + Send,
    WarpFuture: Future<Output = Result<(), EngineError>> + Send + 'static,
{
    crate::profile_reset();
    let selected_warp_count = selection.selected_warp_count(topology);
    let mut mode_state = if inspect_accesses {
        RaceCheckLaunchState::for_topology_and_global_write_allocations(
            topology,
            global_write_allocations,
        )
    } else {
        RaceCheckLaunchState::with_direct_summary_for_topology_and_global_write_allocations(
            topology,
            global_write_allocations,
        )
    };
    mode_state.set_global_memory_model_enabled(global_memory_model_enabled);
    let mode_state = Arc::new(mode_state);
    // Keep one arena reference past the launch so the generated caller's own
    // handles are no longer the last owners; the arena (hundreds of thousands
    // of warp-private allocations) is then freed off the result path.
    let arena_keepalive = physical.clone();
    let execution_started = Instant::now();
    let execution_mode_state = Arc::clone(&mode_state);
    let execution =
        py.detach(move || {
            crate::runtime::launch::run_kernel_engine_launch_report_with_poll_limit::<
                RaceCheckMode,
                _,
                _,
            >(
                physical,
                phase_index,
                execution_mode_state,
                selection,
                max_workers,
                max_polls,
                execution_policy,
                move |warp| make_warp(warp),
            )
        });
    let execution_wall_time = execution_started.elapsed();
    let result_started = Instant::now();
    let result = mode_state.result_for_execution(&execution);
    let result_wall_time = result_started.elapsed();
    #[cfg(feature = "profile")]
    if std::env::var_os("NUMSIM_REPLAY_METRICS").is_some() {
        if let Some(metrics) = mode_state.global_replay_diagnostic_line() {
            eprintln!("TIRX_REPLAY_METRICS {metrics}");
        }
    }
    let cleanup_started = Instant::now();
    crate::defer_drop((mode_state, arena_keepalive));
    let cleanup_wall_time = cleanup_started.elapsed();
    build_native_race_check_phase_result(
        py,
        phase_index,
        phase_name,
        topology,
        result,
        execution,
        topology.warps_per_cluster().min(selected_warp_count),
        selected_warp_count,
        max_polls,
        max_transitions,
        execution_policy.native_loop_iteration_budget(),
        execution_policy.native_loop_reschedule_quantum(),
        execution_wall_time,
        result_wall_time,
        cleanup_wall_time,
    )
}

fn sync_execution_error<'a>(
    result: &RaceCheckResult,
    execution_error: Option<&'a EngineError>,
) -> Option<&'a EngineError> {
    let race_finding_explains_abort = should_filter_race_abort_from_sync(
        result.status(),
        !result.findings().is_empty(),
        result.sync().status(),
        !result.sync().findings().is_empty(),
    );
    if race_finding_explains_abort {
        None
    } else {
        execution_error
    }
}

fn should_filter_race_abort_from_sync(
    race_status: RaceCheckStatus,
    has_race_finding: bool,
    sync_status: SyncCheckStatus,
    has_sync_finding: bool,
) -> bool {
    race_status == RaceCheckStatus::Error
        && has_race_finding
        && sync_status != SyncCheckStatus::Error
        && !has_sync_finding
}

#[allow(clippy::too_many_arguments)]
pub fn build_native_race_check_phase_result(
    py: Python<'_>,
    phase_index: usize,
    phase_name: &str,
    topology: crate::LaunchTopology,
    result: RaceCheckResult,
    execution: ExecutionReport,
    cluster_selected_warp_count: usize,
    selected_warp_count: usize,
    max_polls: Option<usize>,
    max_transitions: Option<usize>,
    native_loop_iteration_budget: usize,
    native_loop_reschedule_quantum: usize,
    execution_wall_time: Duration,
    result_wall_time: Duration,
    cleanup_wall_time: Duration,
) -> PyResult<Py<PyAny>> {
    let payload_started = Instant::now();
    let output = PyDict::new(py);
    output.set_item("schema_version", 3_u32)?;
    output.set_item("execution_model", "direct_online_vc")?;
    let checked_memory_spaces = if result.global_memory_model_enabled() {
        &["global", "shared", "tmem"][..]
    } else {
        &["shared", "tmem"][..]
    };
    output.set_item(
        "checked_memory_spaces",
        PyList::new(py, checked_memory_spaces)?,
    )?;

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
    let execution_incomplete = execution_error.is_some_and(engine_error_is_incomplete)
        || has_cluster_barrier_warp_exit_incomplete(result.sync())
        || partial_domain_setmaxnreg_deadlock;
    let verdict = match result.status() {
        RaceCheckStatus::Error => "error",
        _ if execution_error.is_some() && !execution_incomplete => "error",
        _ if selected_warp_count != topology.warp_count() => "incomplete",
        RaceCheckStatus::Incomplete => "incomplete",
        RaceCheckStatus::Review | RaceCheckStatus::Clean if execution_incomplete => "incomplete",
        RaceCheckStatus::Review => "review",
        RaceCheckStatus::Clean => "clean",
    };
    output.set_item("verdict", verdict)?;
    let report_execution_error = reported_sync_execution_error(result.sync(), execution_error)
        .filter(|_| !partial_domain_setmaxnreg_deadlock);
    output.set_item("findings", findings_to_py(py, &result)?)?;
    output.set_item("advisories", advisories_to_py(py, &result)?)?;
    output.set_item(
        "sync",
        sync_result_to_py(
            py,
            result.sync(),
            sync_execution_error(&result, report_execution_error),
        )?,
    )?;
    output.set_item(
        "incomplete",
        incomplete_to_py(
            py,
            &result,
            report_execution_error,
            (selected_warp_count != topology.warp_count())
                .then_some((selected_warp_count, topology.warp_count())),
        )?,
    )?;
    output.set_item("access_count", result.access_count())?;
    output.set_item("accesses_complete", result.accesses_complete())?;
    output.set_item("accesses", accesses_to_py(py, &result)?)?;

    let execution_profile = crate::profile_snapshot();
    output.set_item(
        "stats",
        execution_stats_to_py(py, Some(&execution.stats), &execution_profile)?,
    )?;
    match report_execution_error {
        Some(error) => output.set_item("execution_error", engine_error_to_py(py, error)?)?,
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
    let payload_wall_time = payload_started.elapsed();
    let execution_cleanup_started = Instant::now();
    drop(execution);
    let execution_cleanup_wall_time = execution_cleanup_started.elapsed();
    let timing = PyDict::new(py);
    timing.set_item(
        "execution_wall_time_us",
        duration_micros(execution_wall_time),
    )?;
    timing.set_item("result_wall_time_us", duration_micros(result_wall_time))?;
    timing.set_item("cleanup_wall_time_us", duration_micros(cleanup_wall_time))?;
    timing.set_item("payload_wall_time_us", duration_micros(payload_wall_time))?;
    timing.set_item(
        "execution_cleanup_wall_time_us",
        duration_micros(execution_cleanup_wall_time),
    )?;
    timing.set_item(
        "total_wall_time_us",
        duration_micros(
            execution_wall_time
                + result_wall_time
                + cleanup_wall_time
                + payload_wall_time
                + execution_cleanup_wall_time,
        ),
    )?;
    output.set_item("timing", timing)?;
    Ok(output.into_any().unbind())
}

fn duration_micros(duration: Duration) -> u64 {
    u64::try_from(duration.as_micros()).unwrap_or(u64::MAX)
}

fn findings_to_py<'py>(py: Python<'py>, result: &RaceCheckResult) -> PyResult<Bound<'py, PyList>> {
    let findings = PyList::empty(py);
    for finding in result.findings() {
        findings.append(finding_to_py(py, finding, result.undeclared_protocol_words())?)?;
    }
    for diagnostic in result.scope_diagnostics() {
        findings.append(scope_mismatch_to_py(py, diagnostic)?)?;
    }
    for diagnostic in result.declared_word_bypasses() {
        findings.append(declared_word_bypass_to_py(py, diagnostic)?)?;
    }
    Ok(findings)
}

fn declared_word_bypass_to_py<'py>(
    py: Python<'py>,
    diagnostic: &DeclaredWordBypassDiagnostic,
) -> PyResult<Bound<'py, PyDict>> {
    let value = PyDict::new(py);
    value.set_item("status", "error")?;
    value.set_item("kind", "signal_protocol_error")?;
    value.set_item(
        "wait_operation",
        dynamic_op_to_py(py, diagnostic.declared_operation())?,
    )?;
    value.set_item(
        "plain_operation",
        dynamic_op_to_py(py, diagnostic.bypassing_operation())?,
    )?;
    value.set_item("wait_warp_id", diagnostic.declared_warp_id())?;
    value.set_item("wait_lane", diagnostic.declared_lane())?;
    value.set_item("plain_warp_id", diagnostic.bypassing_warp_id())?;
    value.set_item("plain_lane", diagnostic.bypassing_lane())?;
    value.set_item("overlap", span_to_py(py, diagnostic.overlap())?)?;
    value.set_item(
        "message",
        format!(
            "Signal {} is accessed by wait_until ({}) and a plain operation ({}) without a happens-before relationship; unsynchronized plain access cannot be mixed with this wait.",
            span_text(diagnostic.overlap()),
            diagnostic.declared_operation(),
            diagnostic.bypassing_operation(),
        ),
    )?;
    value.set_item("hint", "Order initialization/reset and other plain accesses before or after the wait using synchronization. Raw scoped publications remain allowed; not every signal access must use wait_until.")?;
    Ok(value)
}

fn scope_mismatch_to_py<'py>(
    py: Python<'py>,
    diagnostic: &GlobalScopeMismatchDiagnostic,
) -> PyResult<Bound<'py, PyDict>> {
    let value = PyDict::new(py);
    value.set_item("status", "error")?;
    value.set_item("kind", "scope_mismatch")?;
    value.set_item(
        "ordering_domain",
        physical_race_ordering_domain_name(PhysicalRaceOrderingDomain::Memory),
    )?;
    value.set_item("ordering_failure", "scope_mismatch")?;
    value.set_item(
        "release_operation",
        dynamic_op_to_py(py, diagnostic.release_operation())?,
    )?;
    value.set_item(
        "acquire_operation",
        dynamic_op_to_py(py, diagnostic.acquire_operation())?,
    )?;
    value.set_item("release_scope", diagnostic.release_scope().to_string())?;
    value.set_item("acquire_scope", diagnostic.acquire_scope().to_string())?;
    value.set_item("release_warp_id", diagnostic.release_warp_id())?;
    value.set_item("release_lane", diagnostic.release_lane())?;
    value.set_item("acquire_warp_id", diagnostic.acquire_warp_id())?;
    value.set_item("acquire_lane", diagnostic.acquire_lane())?;
    value.set_item("actor_relation", diagnostic.relation().to_string())?;
    value.set_item(
        "message",
        format!(
            "memory accesses {} and {} use scopes .{} and .{}, which do not mutually cover {} actors",
            diagnostic.release_operation(),
            diagnostic.acquire_operation(),
            diagnostic.release_scope(),
            diagnostic.acquire_scope(),
            diagnostic.relation(),
        ),
    )?;
    Ok(value)
}

fn advisories_to_py<'py>(
    py: Python<'py>,
    result: &RaceCheckResult,
) -> PyResult<Bound<'py, PyList>> {
    let advisories = PyList::empty(py);
    for advisory in result.advisories() {
        advisories.append(advisory_to_py(py, advisory)?)?;
    }
    Ok(advisories)
}

fn advisory_to_py<'py>(
    py: Python<'py>,
    advisory: &AliasStaleReadAdvisory,
) -> PyResult<Bound<'py, PyDict>> {
    let value = PyDict::new(py);
    value.set_item("kind", "alias_stale_read")?;
    value.set_item("reader_buffer", advisory.reader_buffer())?;
    value.set_item("writer_buffer", advisory.writer_buffer())?;
    value.set_item("space", advisory.space().to_string())?;
    value.set_item("allocation_id", advisory.allocation().get())?;
    value.set_item(
        "reader_operation",
        dynamic_op_to_py(py, advisory.reader_operation())?,
    )?;
    value.set_item(
        "writer_operation",
        dynamic_op_to_py(py, advisory.writer_operation())?,
    )?;
    value.set_item("occurrences", advisory.occurrences())?;
    let overlaps = PyList::empty(py);
    for span in advisory.overlaps() {
        overlaps.append(span_to_py(py, *span)?)?;
    }
    value.set_item("overlaps", overlaps)?;
    value.set_item(
        "message",
        format!(
            "stale-name read through pool alias: read of '{}' observes physical bytes last written as '{}'",
            advisory.reader_buffer(),
            advisory.writer_buffer(),
        ),
    )?;
    Ok(value)
}

fn finding_to_py<'py>(
    py: Python<'py>,
    finding: &PhysicalRaceFinding,
    protocol_words: &[UndeclaredProtocolWordDiagnostic],
) -> PyResult<Bound<'py, PyDict>> {
    let value = PyDict::new(py);
    let requires_review = finding.requires_unwaited_tmem_load_review();
    value.set_item("status", if requires_review { "review" } else { "error" })?;
    value.set_item("kind", if requires_review { "tmem_lifetime_review" } else { "data_race" })?;
    value.set_item("access_pair", physical_race_kind_name(finding.kind()))?;
    let has_protocol_peer = protocol_words.iter().any(|word| {
        [finding.prior(), finding.current()].iter().any(|access| {
            access.operation().global_warp_id() == word.peer_warp_id()
                && access.lane() == usize::from(word.peer_lane())
        })
    });
    if !requires_review && (has_protocol_peer
        || finding.ordering_failure() == PhysicalRaceOrderingFailure::MissingReleaseAcquire)
    {
        value.set_item("hint",
            "If this access relies on a hand-written spin wait, consider wait_until to express the exit condition. The producer must still provide the required release/synchronization; changing the wait API alone does not establish happens-before.")?;
    }
    value.set_item(
        "ordering_domain",
        physical_race_ordering_domain_name(finding.ordering_failure().domain()),
    )?;
    value.set_item(
        "ordering_failure",
        physical_race_ordering_failure_name(finding.ordering_failure()),
    )?;
    if let PhysicalRaceOrderingFailure::MissingProxyBridge {
        prior_proxy,
        current_proxy,
        prior_domain,
        current_domain,
    } = finding.ordering_failure()
    {
        let details = PyDict::new(py);
        details.set_item("prior_proxy", prior_proxy.to_string())?;
        details.set_item("current_proxy", current_proxy.to_string())?;
        details.set_item(
            "prior_domain",
            physical_race_proxy_domain_name(prior_domain),
        )?;
        details.set_item(
            "current_domain",
            physical_race_proxy_domain_name(current_domain),
        )?;
        value.set_item("proxy_bridge", details)?;
    }
    value.set_item("prior", witness_to_py(py, finding.prior())?)?;
    value.set_item("current", witness_to_py(py, finding.current())?)?;
    value.set_item("overlap", span_to_py(py, finding.overlap())?)?;
    value.set_item(
        "message",
        if requires_review {
            format!(
                "TMEM lifetime conflict requires review because native Racecheck cannot determine whether the earlier tcgen05.ld completed through a true register dependency before the conflicting reuse: {finding}; inspect that dependency, otherwise add tcgen05.wait::ld before reusing the TMEM lifetime; PTX register-dependency rules: {SAME_THREAD_REGISTER_DEPENDENCY_PTX_URL}"
            )
        } else {
            finding.to_string()
        },
    )?;
    Ok(value)
}

const fn physical_race_kind_name(kind: PhysicalRaceKind) -> &'static str {
    match kind {
        PhysicalRaceKind::WriteRead => "write_read",
        PhysicalRaceKind::ReadWrite => "read_write",
        PhysicalRaceKind::WriteWrite => "write_write",
    }
}

const fn physical_race_ordering_domain_name(domain: PhysicalRaceOrderingDomain) -> &'static str {
    match domain {
        PhysicalRaceOrderingDomain::Execution => "execution",
        PhysicalRaceOrderingDomain::Memory => "memory",
        PhysicalRaceOrderingDomain::Completion => "completion",
    }
}

const fn physical_race_ordering_failure_name(failure: PhysicalRaceOrderingFailure) -> &'static str {
    match failure {
        PhysicalRaceOrderingFailure::MissingInterActorSynchronization => "missing_inter_actor_sync",
        PhysicalRaceOrderingFailure::MissingSameWarpLaneOrder => "missing_same_warp_lane_order",
        PhysicalRaceOrderingFailure::MissingReleaseAcquire => "missing_release_acquire",
        PhysicalRaceOrderingFailure::AsyncLifetimeNotDrained => "async_lifetime_not_drained",
        PhysicalRaceOrderingFailure::MissingProxyBridge { .. } => "missing_proxy_bridge",
    }
}

const fn physical_race_proxy_domain_name(domain: PhysicalRaceProxyDomain) -> &'static str {
    match domain {
        PhysicalRaceProxyDomain::Global => "global",
        PhysicalRaceProxyDomain::SharedCta => "shared_cta",
        PhysicalRaceProxyDomain::SharedCluster => "shared_cluster",
        PhysicalRaceProxyDomain::Other => "other",
    }
}

fn witness_to_py<'py>(
    py: Python<'py>,
    witness: &PhysicalRaceWitness,
) -> PyResult<Bound<'py, PyDict>> {
    let value = PyDict::new(py);
    value.set_item("operation", dynamic_op_to_py(py, witness.operation())?)?;
    value.set_item("lane", witness.lane())?;
    value.set_item("access_kind", witness.kind().to_string())?;
    value.set_item("space", witness.space().to_string())?;
    value.set_item("span", span_to_py(py, witness.span())?)?;
    Ok(value)
}

/// `allocation#id[lo..hi)` -- the address text a bypass message reads best with.
fn span_text(span: PhysicalByteSpan) -> String {
    format!(
        "allocation #{}[{}..{})",
        span.allocation().get(),
        span.byte_offset(),
        span.byte_end(),
    )
}

fn span_to_py<'py>(py: Python<'py>, span: PhysicalByteSpan) -> PyResult<Bound<'py, PyDict>> {
    let value = PyDict::new(py);
    value.set_item("allocation_id", span.allocation().get())?;
    value.set_item("byte_offset", span.byte_offset())?;
    value.set_item("byte_len", span.byte_len())?;
    value.set_item("byte_end", span.byte_end())?;
    Ok(value)
}

fn incomplete_to_py<'py>(
    py: Python<'py>,
    result: &RaceCheckResult,
    execution_error: Option<&EngineError>,
    subset_execution: Option<(usize, usize)>,
) -> PyResult<Bound<'py, PyList>> {
    let incomplete = PyList::empty(py);
    for reason in result.incomplete_reasons() {
        incomplete.append(race_incomplete_to_py(py, reason)?)?;
    }
    append_cluster_sync_incomplete(py, &incomplete, result, &mut Vec::new())?;
    if let Some(error) = execution_error.filter(|error| engine_error_is_incomplete(error)) {
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

fn race_incomplete_to_py<'py>(
    py: Python<'py>,
    reason: &RaceCheckIncompleteReason,
) -> PyResult<Bound<'py, PyDict>> {
    let value = PyDict::new(py);
    value.set_item("kind", "analysis_incomplete")?;
    match reason {
        RaceCheckIncompleteReason::GlobalWriteSeedIncomplete => {
            value.set_item("reason", "global_write_seed_incomplete")?;
            value.set_item("message", "A write reached an allocation outside the initial write set; repeat from the original inputs with all global allocations to retain preceding reads.")?;
        }
        RaceCheckIncompleteReason::AnalysisGap {
            operation,
            kind,
        } => {
            value.set_item(
                "reason",
                match kind.domain() {
                    AnalysisGapDomain::Tcgen => "tcgen_access_unmodeled",
                    AnalysisGapDomain::ClusterBarrier => "cluster_barrier_unaligned_unmodeled",
                    AnalysisGapDomain::Atomic => "atomic_lane_serialization_unmodeled",
                },
            )?;
            value.set_item("operation", dynamic_op_to_py(py, operation)?)?;
            value.set_item("domain", kind.domain().name())?;
            value.set_item("effect", kind.name())?;
        }
        RaceCheckIncompleteReason::EffectCommitUnobserved { operation, effect } => {
            value.set_item("reason", "effect_commit_unobserved")?;
            value.set_item("operation", dynamic_op_to_py(py, operation)?)?;
            value.set_item("effect", effect)?;
        }
        RaceCheckIncompleteReason::BarrierGenerationUnavailable {
            operation,
            barrier_id,
        } => {
            value.set_item("reason", "barrier_generation_unavailable")?;
            value.set_item("operation", dynamic_op_to_py(py, operation)?)?;
            value.set_item("barrier", physical_barrier_to_py(py, *barrier_id)?)?;
        }
        RaceCheckIncompleteReason::BarrierPayloadUnavailable {
            operation,
            barrier_id,
            generation,
        } => {
            value.set_item("reason", "barrier_payload_unavailable")?;
            value.set_item("operation", dynamic_op_to_py(py, operation)?)?;
            value.set_item("barrier", physical_barrier_to_py(py, *barrier_id)?)?;
            value.set_item("generation", generation)?;
        }
        RaceCheckIncompleteReason::DeclaredWordWaitUnexplained { operation } => {
            value.set_item("reason", "wait_exit_unproven")?;
            value.set_item("message", "Cannot verify why wait_until completed: the available write history, already-ordered values, and launch value do not explain its exit. This is not proof of a kernel error or deadlock.")?;
            value.set_item("hint", "Check that the write satisfying the wait is supported and recorded by the checker; retain this case when reporting an analysis coverage gap.")?;
            value.set_item("operation", dynamic_op_to_py(py, operation)?)?;
        }
        RaceCheckIncompleteReason::DeclaredWordHistoryTruncated { operation } => {
            value.set_item("reason", "signal_history_truncated")?;
            value.set_item("message", "Cannot finish signal analysis because the retained write history reached its limit.")?;
            value.set_item("hint", "Use a smaller reproducer or report the history limit; this result cannot certify the kernel.")?;
            value.set_item("operation", dynamic_op_to_py(py, operation)?)?;
        }
        RaceCheckIncompleteReason::DeclaredWordWriteUnrecorded { operation } => {
            value.set_item("reason", "signal_write_not_recorded")?;
            value.set_item("message", "Cannot finish signal analysis because an overlapping write was not recorded in the supported signal history.")?;
            value.set_item("hint", "Include this write operation when reporting the checker coverage gap; this result is not a proven kernel error.")?;
            value.set_item("operation", dynamic_op_to_py(py, operation)?)?;
        }
        RaceCheckIncompleteReason::RetiredRecordsUnobserved { operation } => {
            value.set_item("reason", "retired_records_unobserved")?;
            value.set_item("operation", dynamic_op_to_py(py, operation)?)?;
        }
        RaceCheckIncompleteReason::FindingsTruncated { retained, dropped } => {
            value.set_item("reason", "findings_truncated")?;
            value.set_item("retained", retained)?;
            value.set_item("dropped", dropped)?;
        }
        RaceCheckIncompleteReason::ShadowRejected { operation, reason } => {
            value.set_item("reason", "shadow_rejected")?;
            value.set_item("operation", dynamic_op_to_py(py, operation)?)?;
            value.set_item("cause", reason)?;
        }
        RaceCheckIncompleteReason::AsyncPayloadAccessUnmodeled { operation } => {
            value.set_item("reason", "async_payload_access_unmodeled")?;
            value.set_item("operation", dynamic_op_to_py(py, operation)?)?;
        }
        RaceCheckIncompleteReason::GlobalMemoryModelUnsupported {
            operation,
            kind,
            reason,
        } => {
            value.set_item("reason", kind)?;
            value.set_item("operation", dynamic_op_to_py(py, operation)?)?;
            value.set_item("domain", "scoped_global_memory")?;
            value.set_item("cause", reason)?;
        }
    }
    Ok(value)
}

fn append_cluster_sync_incomplete<'py>(
    py: Python<'py>,
    incomplete: &Bound<'py, PyList>,
    result: &RaceCheckResult,
    seen: &mut Vec<SyncCheckIncompleteReason>,
) -> PyResult<()> {
    for reason in result.sync().incomplete_reasons() {
        if matches!(
            reason,
            SyncCheckIncompleteReason::ClusterBarrierParticipantExitUnmodeled { .. }
                | SyncCheckIncompleteReason::ClusterBarrierRearrivalWithoutWaitUnmodeled { .. }
        ) && !seen.contains(reason)
        {
            incomplete.append(sync_incomplete_to_py(py, reason)?)?;
            seen.push(reason.clone());
        }
    }
    Ok(())
}

fn accesses_to_py<'py>(py: Python<'py>, result: &RaceCheckResult) -> PyResult<Bound<'py, PyList>> {
    let accesses = PyList::empty(py);
    for access in result.accesses() {
        accesses.append(access_to_py(
            py,
            access,
            result.global_memory_model_enabled(),
        )?)?;
    }
    Ok(accesses)
}

fn access_to_py<'py>(
    py: Python<'py>,
    access: &RaceCheckAccessRecord,
    include_memory_semantics: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let value = PyDict::new(py);
    value.set_item("operation", dynamic_op_to_py(py, access.operation())?)?;
    value.set_item("access_kind", access.descriptor().kind().to_string())?;
    value.set_item("space", access.descriptor().space().to_string())?;
    match access.logical_buffer() {
        Some(logical_buffer) => value.set_item("logical_buffer", logical_buffer)?,
        None => value.set_item("logical_buffer", py.None())?,
    }
    value.set_item("width", access.descriptor().width().bytes())?;
    if include_memory_semantics {
        let semantics = access.descriptor().memory_semantics();
        value.set_item("memory_order", semantics.order().to_string())?;
        match semantics.scope() {
            Some(scope) => value.set_item("memory_scope", scope.to_string())?,
            None => value.set_item("memory_scope", py.None())?,
        }
        value.set_item("memory_proxy", semantics.proxy().to_string())?;
        value.set_item("memory_access_class", semantics.class().to_string())?;
        // Whether this access belongs to a word the kernel declared through
        // a declared wait. A declared wait adjudicates no access at
        // all -- the polling is the engine's -- so the journal shows only the
        // protocol's stores and contributions, and this is what marks them.
    }
    value.set_item("active_lane_count", access.active_lane_count())?;
    let lanes = PyList::empty(py);
    for lane in access.lanes() {
        lanes.append(lane_access_to_py(py, lane)?)?;
    }
    value.set_item("lanes", lanes)?;
    Ok(value)
}

fn lane_access_to_py<'py>(
    py: Python<'py>,
    access: &LanePhysicalAccess,
) -> PyResult<Bound<'py, PyDict>> {
    let value = PyDict::new(py);
    value.set_item("lane", access.provenance().lane())?;
    let spans = PyList::empty(py);
    for span in access.footprint().spans() {
        spans.append(span_to_py(py, *span)?)?;
    }
    value.set_item("spans", spans)?;
    Ok(value)
}

#[cfg(test)]
mod tests {
    use crate::{RaceCheckStatus, SyncCheckStatus};

    use super::should_filter_race_abort_from_sync;

    #[test]
    fn race_abort_filter_preserves_natural_sync_incomplete() {
        assert!(should_filter_race_abort_from_sync(
            RaceCheckStatus::Error,
            true,
            SyncCheckStatus::Incomplete,
            false,
        ));
    }

    #[test]
    fn race_abort_filter_never_hides_a_sync_error_or_finding() {
        assert!(!should_filter_race_abort_from_sync(
            RaceCheckStatus::Error,
            true,
            SyncCheckStatus::Error,
            true,
        ));
        assert!(!should_filter_race_abort_from_sync(
            RaceCheckStatus::Error,
            true,
            SyncCheckStatus::Incomplete,
            true,
        ));
        assert!(!should_filter_race_abort_from_sync(
            RaceCheckStatus::Incomplete,
            false,
            SyncCheckStatus::Incomplete,
            false,
        ));
    }
}
