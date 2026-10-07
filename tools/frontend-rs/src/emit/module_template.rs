//! Host/artifact template with `__NUMSIM_*__` placeholders. Native emission
//! binds module contents; `build.py` later binds the cache key and module name.

pub const MODULE_TEMPLATE: &str = r#"#![allow(warnings)]
#![feature(optimize_attribute)]

__NUMSIM_ENGINE_ABI_IMPORTS__
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyModule};
use std::sync::Arc;


macro_rules! numsim_local_future {
($vis:vis) => {
$vis struct NumSimModuleFuture<F>($vis F);

impl<F: std::future::Future> std::future::Future for NumSimModuleFuture<F> {
    type Output = F::Output;

    fn poll(
        self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Self::Output> {
        // SAFETY: pinning the wrapper also pins its only field, and this
        // projection never moves that field.
        unsafe { self.map_unchecked_mut(|wrapper| &mut wrapper.0) }.poll(cx)
    }
}
};
}
numsim_local_future!(pub(crate));

__NUMSIM_MEMORY_HELPERS__

const CACHE_KEY: &str = "__NUMSIM_CACHE_KEY__";
const NUMSIM_ABI_VERSION: u32 = __NUMSIM_ABI_VERSION__;
const ENGINE_HASH: &str = env!("NUMSIM_ARTIFACT_ENGINE_HASH");
const BUILD_IDENTITY_JSON: &str = env!("NUMSIM_ARTIFACT_BUILD_IDENTITY_JSON");

type NumSimFnMap<S> = FrontendFnMap<S>;
type NumSimGlobalMap = FrontendTableMap<v2::Global>;
type NumSimSharedMap = FrontendTableMap<v2::Shared>;
type NumSimLocalMap = FrontendTableMap<v2::Local>;
type NumSimRegisterMap = FrontendTableMap<v2::Register>;
type NumSimTmemMap = FrontendTableMap<v2::Tmem>;
const NumSimGlobalMap: NumSimGlobalMap = FrontendTableMap::new();
const NumSimSharedMap: NumSimSharedMap = FrontendTableMap::new();
const NumSimLocalMap: NumSimLocalMap = FrontendTableMap::new();
const NumSimRegisterMap: NumSimRegisterMap = FrontendTableMap::new();
const NumSimTmemMap: NumSimTmemMap = FrontendTableMap::new();

__NUMSIM_KERNEL_STRUCTS__



__NUMSIM_KERNEL_FUNCTIONS__

#[pyfunction]
fn metadata(py: Python<'_>) -> PyResult<Py<PyAny>> {
    build_artifact_metadata_with_build_identity(
        py,
        CACHE_KEY,
        NUMSIM_ABI_VERSION,
        ENGINE_HASH,
        BUILD_IDENTITY_JSON,
        &[__NUMSIM_KERNEL_NAMES__],
        &[__NUMSIM_TOPOLOGY_DIMENSIONS__],
    )
}

// numsim-numeric-entrypoints:begin
fn run_impl(
    py: Python<'_>,
    inputs: &Bound<'_, PyDict>,
    subset: Option<&Bound<'_, PyAny>>,
    max_workers: usize,
    native_loop_iteration_budget: Option<usize>,
    native_loop_reschedule_quantum: Option<usize>,
    selected_phase: Option<usize>,
    include_allocation_state: bool,
) -> PyResult<Py<PyAny>> {
    if max_workers == 0 {
        return Err(PyValueError::new_err(
            "NumSim max_workers must be positive",
        ));
    }
    let execution_policy = ExecutionPolicy::new(
        native_loop_iteration_budget.unwrap_or(DEFAULT_NATIVE_LOOP_ITERATION_BUDGET),
        native_loop_reschedule_quantum.unwrap_or(DEFAULT_NATIVE_LOOP_RESCHEDULE_QUANTUM),
    )
    .map_err(|error| PyValueError::new_err(error.to_string()))?;
    if let Some(phase_index) = selected_phase {
        if phase_index >= __NUMSIM_KERNEL_COUNT___usize {
            return Err(PyValueError::new_err(format!(
                "NumSim selected phase {phase_index} is outside [0, __NUMSIM_KERNEL_COUNT__)",
            )));
        }
    }
    let global = GlobalMemory::new_reviewing_uninitialized_reads();
    let allocation_ids = extract_allocations(inputs, &global, NUMSIM_ABI_VERSION)?;
    let output_allocations = extract_output_allocations(inputs, allocation_ids.len())?;
    let mut run_result = RunResultBuilder::new();
__NUMSIM_RUN_PHASES__
    build_run_result(
        py,
        inputs,
        run_result,
        &global,
        &allocation_ids,
        &output_allocations,
        include_allocation_state,
    )
}

#[pyfunction(signature = (
    inputs,
    subset=None,
    max_workers=1,
    native_loop_iteration_budget=None,
    native_loop_reschedule_quantum=None,
))]
fn run(
    py: Python<'_>,
    inputs: &Bound<'_, PyDict>,
    subset: Option<&Bound<'_, PyAny>>,
    max_workers: usize,
    native_loop_iteration_budget: Option<usize>,
    native_loop_reschedule_quantum: Option<usize>,
) -> PyResult<Py<PyAny>> {
    run_impl(
        py,
        inputs,
        subset,
        max_workers,
        native_loop_iteration_budget,
        native_loop_reschedule_quantum,
        None,
        false,
    )
}
#[pyfunction(name = "_advance_phase", signature = (
    inputs,
    phase_index,
    subset=None,
    max_workers=1,
    native_loop_iteration_budget=None,
    native_loop_reschedule_quantum=None,
))]
fn advance_phase(
    py: Python<'_>,
    inputs: &Bound<'_, PyDict>,
    phase_index: usize,
    subset: Option<&Bound<'_, PyAny>>,
    max_workers: usize,
    native_loop_iteration_budget: Option<usize>,
    native_loop_reschedule_quantum: Option<usize>,
) -> PyResult<Py<PyAny>> {
    run_impl(
        py,
        inputs,
        subset,
        max_workers,
        native_loop_iteration_budget,
        native_loop_reschedule_quantum,
        Some(phase_index),
        true,
    )
}
// numsim-numeric-entrypoints:end

// numsim-analysis-entrypoints:begin
// numsim-synccheck-entrypoint:begin
#[pyfunction(name = "_native_synccheck_phase", signature = (
    inputs,
    phase_index,
    max_warp_preemptions,
    max_completion_schedule_deviations,
    max_schedules,
    max_backtrack_nodes,
    max_events_per_run,
    max_total_events,
    max_loop_steps,
    max_wall_time_ms,
    max_diagnostic_bytes,
    subset=None,
    max_workers=1,
    max_polls=None,
    max_transitions=None,
    native_loop_iteration_budget=None,
    native_loop_reschedule_quantum=None,
))]
fn native_synccheck_phase(
    py: Python<'_>,
    inputs: &Bound<'_, PyDict>,
    phase_index: usize,
    max_warp_preemptions: u64,
    max_completion_schedule_deviations: u64,
    max_schedules: u64,
    max_backtrack_nodes: u64,
    max_events_per_run: u64,
    max_total_events: u64,
    max_loop_steps: u64,
    max_wall_time_ms: u64,
    max_diagnostic_bytes: u64,
    subset: Option<&Bound<'_, PyAny>>,
    max_workers: usize,
    max_polls: Option<usize>,
    max_transitions: Option<usize>,
    native_loop_iteration_budget: Option<usize>,
    native_loop_reschedule_quantum: Option<usize>,
) -> PyResult<Py<PyAny>> {
    if phase_index >= __NUMSIM_KERNEL_COUNT___usize {
        return Err(PyValueError::new_err(format!(
            "native synccheck phase {phase_index} is outside [0, __NUMSIM_KERNEL_COUNT__)",
        )));
    }
    if max_workers == 0 {
        return Err(PyValueError::new_err(
            "native synccheck max_workers must be positive",
        ));
    }
    let execution_policy = ExecutionPolicy::new(
        native_loop_iteration_budget.unwrap_or(DEFAULT_NATIVE_LOOP_ITERATION_BUDGET),
        native_loop_reschedule_quantum.unwrap_or(DEFAULT_NATIVE_LOOP_RESCHEDULE_QUANTUM),
    )
    .map_err(|error| PyValueError::new_err(error.to_string()))?;
    let global = GlobalMemory::new_reviewing_uninitialized_reads();
    let allocation_ids = extract_allocations(inputs, &global, NUMSIM_ABI_VERSION)?;
    match phase_index {
__NUMSIM_NATIVE_SYNC_CHECK_PHASES__
        _ => unreachable!("phase index was validated"),
    }
}
// numsim-synccheck-entrypoint:end

// numsim-racecheck-entrypoint:begin
#[pyfunction(name = "_native_racecheck_phase", signature = (
    inputs,
    phase_index,
    subset=None,
    inspect_accesses=false,
    max_workers=1,
    max_polls=None,
    max_transitions=None,
    native_loop_iteration_budget=None,
    native_loop_reschedule_quantum=None,
))]
fn native_racecheck_phase(
    py: Python<'_>,
    inputs: &Bound<'_, PyDict>,
    phase_index: usize,
    subset: Option<&Bound<'_, PyAny>>,
    inspect_accesses: bool,
    max_workers: usize,
    max_polls: Option<usize>,
    max_transitions: Option<usize>,
    native_loop_iteration_budget: Option<usize>,
    native_loop_reschedule_quantum: Option<usize>,
) -> PyResult<Py<PyAny>> {
    if phase_index >= __NUMSIM_KERNEL_COUNT___usize {
        return Err(PyValueError::new_err(format!(
            "native racecheck phase {phase_index} is outside [0, __NUMSIM_KERNEL_COUNT__)",
        )));
    }
    if max_workers == 0 {
        return Err(PyValueError::new_err(
            "native racecheck max_workers must be positive",
        ));
    }
    if max_polls == Some(0) {
        return Err(PyValueError::new_err(
            "native racecheck max_polls must be positive",
        ));
    }
    if max_transitions == Some(0) {
        return Err(PyValueError::new_err(
            "native racecheck max_transitions must be positive",
        ));
    }
    let execution_policy = ExecutionPolicy::new(
        native_loop_iteration_budget.unwrap_or(DEFAULT_NATIVE_LOOP_ITERATION_BUDGET),
        native_loop_reschedule_quantum.unwrap_or(DEFAULT_NATIVE_LOOP_RESCHEDULE_QUANTUM),
    )
    .map_err(|error| PyValueError::new_err(error.to_string()))?;
    with_complete_global_write_seed(py, |full_global_write_seed| {
        let global = GlobalMemory::new_reviewing_uninitialized_reads();
        let allocation_ids = extract_allocations(inputs, &global, NUMSIM_ABI_VERSION)?;
        match phase_index {
__NUMSIM_NATIVE_RACE_CHECK_PHASES__
            _ => unreachable!("phase index was validated"),
        }
    })
}
// numsim-racecheck-entrypoint:end

// numsim-analysis-entrypoints:end

#[pymodule]
fn __NUMSIM_MODULE__(module: &Bound<'_, PyModule>) -> PyResult<()> {
    prime_host_load_sample();
__NUMSIM_EXPORTS__
    Ok(())
}
"#;
