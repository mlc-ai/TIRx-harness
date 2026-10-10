"""Public NumSim API."""

from __future__ import annotations

import copy
import os
from collections.abc import Iterable, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from pathlib import Path
from threading import Lock
from typing import Any, Literal

import numpy as np
from threadpoolctl import threadpool_limits

from .abi import abi_metadata
from .bindings import (
    MulticastWindow,
    PreparedBindings,
    SymmetricBuffer,
    prepare_bindings,
    prepare_rank_bindings,
    rank_binding_name,
)
from .cases import (
    ComparisonRegion,
    ComparisonSpec,
    ExecutionAssumptions,
    NumSimCase,
)
from .errors import NumSimBuildError, NumSimExecutionError
from .host_abi import HostAbiContract, HostAbiError, HostBindingSlot, build_host_abi
from .report import Mismatch, NumSimReport
from .transpiler import native_frontend
from .transpiler.frontend import (
    ModuleSpec,
    attach_source_nodes,
)
from .transpiler.semantic_ir import cache_semantic_ir_serializations

_DEFAULT_NATIVE_LOOP_ITERATION_BUDGET = 1_000_000
_DEFAULT_NATIVE_LOOP_RESCHEDULE_QUANTUM = 64
_MAX_U64 = (1 << 64) - 1


def _nonnegative_u64(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} must be nonnegative")
    if value > _MAX_U64:
        raise ValueError(f"{name} must fit in an unsigned 64-bit integer")
    return value


def _positive_u64(name: str, value: int) -> int:
    value = _nonnegative_u64(name, value)
    if value == 0:
        raise ValueError(f"{name} must be positive")
    return value


@dataclass(frozen=True)
class ExecutionSubset:
    """Select launch domains for execution.

    ``cluster_ids`` are cluster coordinates. ``cta_ids`` are flattened global
    CTA IDs and must select a union of complete clusters; a single CTA is a
    complete cluster when ``ctas_per_cluster == 1``. When both are present,
    NumSim executes their intersection.
    """

    cluster_ids: tuple[int, ...] | list[int] | None = None
    cta_ids: tuple[int, ...] | list[int] | None = None

    def to_payload(self) -> dict[str, list[int] | None]:
        return {
            "cluster_ids": None if self.cluster_ids is None else list(self.cluster_ids),
            "cta_ids": None if self.cta_ids is None else list(self.cta_ids),
        }


ExecutionSubsetSelection = ExecutionSubset | Mapping[int, ExecutionSubset]


@dataclass(frozen=True)
class CoverageBounds:
    max_warp_preemptions: int
    max_completion_schedule_deviations: int

    def __post_init__(self) -> None:
        for name in ("max_warp_preemptions", "max_completion_schedule_deviations"):
            _nonnegative_u64(f"CoverageBounds.{name}", getattr(self, name))

    def to_payload(self) -> dict[str, int]:
        return {
            "max_warp_preemptions": self.max_warp_preemptions,
            "max_completion_schedule_deviations": self.max_completion_schedule_deviations,
        }


@dataclass(frozen=True)
class ResourceLimits:
    max_schedules: int
    max_backtrack_nodes: int
    max_events_per_run: int
    max_total_events: int
    max_loop_steps: int
    max_wall_time_ms: int
    max_diagnostic_bytes: int

    def __post_init__(self) -> None:
        for name in (
            "max_schedules",
            "max_backtrack_nodes",
            "max_events_per_run",
            "max_total_events",
            "max_loop_steps",
            "max_wall_time_ms",
            "max_diagnostic_bytes",
        ):
            _positive_u64(f"ResourceLimits.{name}", getattr(self, name))

    def to_payload(self) -> dict[str, int]:
        return {
            "max_schedules": self.max_schedules,
            "max_backtrack_nodes": self.max_backtrack_nodes,
            "max_events_per_run": self.max_events_per_run,
            "max_total_events": self.max_total_events,
            "max_loop_steps": self.max_loop_steps,
            "max_wall_time_ms": self.max_wall_time_ms,
            "max_diagnostic_bytes": self.max_diagnostic_bytes,
        }


def default_coverage_bounds() -> CoverageBounds:
    """Return the single checker default for schedule-exploration coverage.

    Every entry point that runs a checker phase without explicit bounds -- the
    ``Engine`` phase methods and ``checker_runner`` -- resolves from here, so a
    direct phase caller cannot silently search a smaller space than production.
    """

    return CoverageBounds(max_warp_preemptions=2, max_completion_schedule_deviations=2)


def default_resource_limits() -> ResourceLimits:
    """Return the single checker default for one bounded analysis run."""

    return ResourceLimits(
        max_schedules=10_000,
        max_backtrack_nodes=100_000,
        max_events_per_run=1_000_000,
        max_total_events=10_000_000,
        max_loop_steps=10_000_000,
        max_wall_time_ms=30_000,
        max_diagnostic_bytes=16 * 1024 * 1024,
    )


@dataclass(frozen=True)
class NativeAnalysisResult:
    checker: Literal["synccheck", "racecheck"]
    payload: dict[str, Any]

    @property
    def verdict(self) -> str:
        return str(self.payload["verdict"])

    @property
    def findings(self) -> list[dict[str, Any]]:
        return list(self.payload.get("findings", ()))

    @property
    def advisories(self) -> list[dict[str, Any]]:
        return list(self.payload.get("advisories", ()))

    @property
    def incomplete(self) -> list[dict[str, Any]]:
        return list(self.payload.get("incomplete", ()))

    @property
    def coverage(self) -> dict[str, Any] | None:
        value = self.payload.get("coverage")
        return None if value is None else dict(value)

    @property
    def counterexample(self) -> dict[str, Any] | None:
        value = self.payload.get("counterexample")
        return None if value is None else dict(value)

    def to_dict(self) -> dict[str, Any]:
        return copy.deepcopy(self.payload)


def _external_grid_dependency_phases(spec: ModuleSpec) -> tuple[int, ...]:
    requirement = native_frontend.registry()["external_grid_dependency_requirement"]
    return tuple(
        index
        for index, kernel in enumerate(spec.kernels)
        if requirement in kernel.semantic_requirements
    )


def _execution_assumptions_payload(
    spec: ModuleSpec, assumptions: ExecutionAssumptions | None
) -> dict[str, list[int]]:
    required = set(_external_grid_dependency_phases(spec))
    if assumptions is None:
        provided: set[int] = set()
    elif isinstance(assumptions, ExecutionAssumptions):
        values = assumptions.external_grid_dependencies_satisfied
        provided = set(values)
        for value in values:
            if value >= len(spec.kernels):
                raise ValueError(
                    f"external grid dependency phase {value} is outside [0, {len(spec.kernels)})"
                )
    else:
        raise TypeError("NumSim assumptions must be ExecutionAssumptions or None")
    unexpected = sorted(provided - required)
    if unexpected:
        raise ValueError(
            "external grid dependency assumptions were supplied for phases without "
            f"griddepcontrol.wait: {unexpected}"
        )
    # A standalone NumSim launch cannot observe the scheduler or producer
    # kernel that makes an external grid dependency ready. Treat that launch
    # boundary as satisfied by definition. The artifact executes phases in
    # sequence, so internal prerequisites have completed before a phase starts,
    # regardless of whether any thread issued the optional launch hint.
    return {"external_grid_dependencies_satisfied": sorted(required)}


def _normalized_execution_subset_payload(
    subset: ExecutionSubset, *, field: str
) -> dict[str, list[int] | None]:
    if not isinstance(subset, ExecutionSubset):
        raise TypeError(f"{field} must be an ExecutionSubset")
    payload: dict[str, list[int] | None] = {}
    for name in ("cluster_ids", "cta_ids"):
        values = getattr(subset, name)
        if values is None:
            payload[name] = None
            continue
        if not isinstance(values, tuple | list):
            raise TypeError(f"{field}.{name} must be a tuple, list, or None")
        normalized: list[int] = []
        for value in values:
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{field}.{name} must contain integers")
            if value < 0:
                raise ValueError(f"{field}.{name} contains negative ID {value}")
            normalized.append(value)
        if len(normalized) != len(set(normalized)):
            raise ValueError(f"{field}.{name} contains duplicate IDs")
        payload[name] = sorted(normalized)
    return payload


def _execution_subset_payload(
    subset: ExecutionSubsetSelection | None, *, kernel_count: int
) -> dict[str, list[int] | None] | list[dict[str, list[int] | None] | None] | None:
    """Freeze a launch selection without broadcasting it across kernel phases."""

    if kernel_count <= 0:
        raise ValueError("NumSim execution subset requires at least one kernel phase")
    if subset is None:
        return None
    if isinstance(subset, ExecutionSubset):
        if kernel_count != 1:
            raise ValueError(
                "a multi-kernel NumSim launch requires subset={phase_index: "
                "ExecutionSubset(...)}; one subset is not broadcast across phases"
            )
        return _normalized_execution_subset_payload(subset, field="NumSim subset")
    if not isinstance(subset, Mapping):
        raise TypeError("NumSim subset must be an ExecutionSubset, a phase-index mapping, or None")
    phases: list[dict[str, list[int] | None] | None] = [None] * kernel_count
    for phase_index, phase_subset in subset.items():
        if isinstance(phase_index, bool) or not isinstance(phase_index, int):
            raise TypeError("NumSim subset phase indices must be integers")
        if phase_index < 0 or phase_index >= kernel_count:
            raise ValueError(
                f"NumSim subset phase index {phase_index} is outside [0, {kernel_count})"
            )
        phases[phase_index] = _normalized_execution_subset_payload(
            phase_subset, field=f"NumSim subset phase {phase_index}"
        )
    return phases


@dataclass
class NumSimResult:
    outputs: dict[str, Any]
    diagnostics: list[dict[str, Any]] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)

    @property
    def verdict(self) -> str:
        if any(item.get("status") == "review" for item in self.diagnostics):
            return "review"
        return "clean"

    def assert_close(
        self, expected: dict[str, Any], tolerances: dict[str, ComparisonSpec] | None = None
    ) -> None:
        compare(self, expected, tolerances=tolerances).require_ok()


@dataclass(frozen=True)
class CompiledModule:
    spec: ModuleSpec
    artifact: Any
    source: Any = field(default=None, repr=False, compare=False)
    cache_dir: Path | None = field(default=None, repr=False, compare=False)
    _generated_opt_level: int = field(default=3, repr=False, compare=False)
    _analysis_capable: bool = field(default=False, repr=False, compare=False)
    _analysis_checker: Literal["synccheck", "racecheck"] | None = field(
        default=None, repr=False, compare=False
    )

    @property
    def cache_key(self) -> str:
        return self.artifact.key

    @property
    def rust_source(self) -> str:
        return self.artifact.source

    @property
    def library_path(self) -> Path:
        return self.artifact.library_path

    def load(self):
        return self.artifact.load()

@dataclass(frozen=True)
class _PreparedExecution:
    bindings: PreparedBindings
    output_names: frozenset[str]
    external_names: dict[str, str]
    assumptions: dict[str, list[int]]

    @property
    def selected_output_names(self) -> frozenset[str]:
        return frozenset(self.external_names.values())


@dataclass(frozen=True)
class _ExecutionPolicy:
    native_loop_iteration_budget: int = _DEFAULT_NATIVE_LOOP_ITERATION_BUDGET
    native_loop_reschedule_quantum: int = _DEFAULT_NATIVE_LOOP_RESCHEDULE_QUANTUM


def _native_metadata_for_execution(module: CompiledModule, native: Any) -> dict[str, Any]:
    try:
        metadata = native.metadata()
    except Exception as error:
        raise NumSimExecutionError(
            f"NumSim artifact rejected its metadata during execution: "
            f"{type(error).__name__}: {error}"
        ) from error
    if not isinstance(metadata, dict):
        raise NumSimExecutionError(
            f"NumSim artifact metadata has type {type(metadata).__name__}, expected dict"
        )
    validator = getattr(module.artifact, "validate_native_metadata", None)
    if validator is not None:
        try:
            validator(metadata)
        except NumSimBuildError as error:
            raise NumSimExecutionError(
                f"NumSim artifact metadata changed before execution: {error}"
            ) from error
    return metadata


def _positive_policy_integer(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"NumSim {name} must be a positive integer")
    if value <= 0:
        raise ValueError(f"NumSim {name} must be positive")
    return value


# BLAS pools are process-wide; all overlapping native executions share one
# temporary limit. Rust workers own parallelism, including NumPy-backed GEMMs.
_blas_limit_lock = Lock()
_blas_active_calls = 0
_blas_limit = None


@contextmanager
def _blas_thread_context():
    global _blas_active_calls, _blas_limit
    with _blas_limit_lock:
        if _blas_active_calls == 0:
            _blas_limit = threadpool_limits(limits=1, user_api="blas")
        _blas_active_calls += 1
    try:
        yield
    finally:
        with _blas_limit_lock:
            _blas_active_calls -= 1
            if _blas_active_calls == 0:
                _blas_limit.restore_original_limits()
                _blas_limit = None


class Engine:
    """Python facade for a generated artifact's Rust-owned EngineState."""

    def __init__(
        self,
        max_workers: int | Literal["auto"] = 8,
        *,
        native_loop_iteration_budget: int = _DEFAULT_NATIVE_LOOP_ITERATION_BUDGET,
        native_loop_reschedule_quantum: int = _DEFAULT_NATIVE_LOOP_RESCHEDULE_QUANTUM,
    ) -> None:
        if max_workers == "auto":
            resolved = os.cpu_count() or 1
        elif isinstance(max_workers, bool) or not isinstance(max_workers, int):
            raise TypeError("NumSim max_workers must be a positive integer or 'auto'")
        else:
            resolved = max_workers
        if resolved <= 0:
            raise ValueError("NumSim max_workers must be positive")
        self._max_workers = resolved
        self._execution_policy = _ExecutionPolicy(
            native_loop_iteration_budget=_positive_policy_integer(
                "native_loop_iteration_budget", native_loop_iteration_budget
            ),
            native_loop_reschedule_quantum=_positive_policy_integer(
                "native_loop_reschedule_quantum", native_loop_reschedule_quantum
            ),
        )

    @property
    def max_workers(self) -> int:
        return self._max_workers

    @property
    def native_loop_iteration_budget(self) -> int:
        return self._execution_policy.native_loop_iteration_budget

    @property
    def native_loop_reschedule_quantum(self) -> int:
        return self._execution_policy.native_loop_reschedule_quantum

    def run(
        self,
        module: CompiledModule,
        inputs: dict[str, Any],
        *,
        subset: ExecutionSubsetSelection | None = None,
        assumptions: ExecutionAssumptions | None = None,
        outputs: Iterable[str] | Mapping[str, str] | None = None,
    ) -> NumSimResult:
        execution = self._prepare_execution(
            module,
            inputs,
            outputs=outputs,
            assumptions=assumptions,
        )
        return self._execute_prepared(module, execution, subset=subset)

    def run_synccheck_phase(
        self,
        module: CompiledModule,
        inputs: dict[str, Any],
        *,
        phase_index: int = 0,
        subset: ExecutionSubset | None = None,
        assumptions: ExecutionAssumptions | None = None,
        coverage_bounds: CoverageBounds | None = None,
        resource_limits: ResourceLimits | None = None,
        max_polls: int | None = None,
        max_transitions: int | None = None,
        advance_prefix: bool = False,
        _prepared_bindings: PreparedBindings | None = None,
    ) -> NativeAnalysisResult:
        if not isinstance(advance_prefix, bool):
            raise TypeError("native synccheck advance_prefix must be a bool")
        return self._run_native_synccheck_phase(
            module,
            inputs,
            coverage_bounds=(
                default_coverage_bounds() if coverage_bounds is None else coverage_bounds
            ),
            resource_limits=(
                default_resource_limits() if resource_limits is None else resource_limits
            ),
            phase_index=phase_index,
            subset=subset,
            assumptions=assumptions,
            max_polls=max_polls,
            max_transitions=max_transitions,
            advance_prefix=advance_prefix,
            prepared_bindings=_prepared_bindings,
        )

    def run_racecheck_phase(
        self,
        module: CompiledModule,
        inputs: dict[str, Any],
        *,
        phase_index: int = 0,
        subset: ExecutionSubset | None = None,
        inspect_accesses: bool = False,
        max_polls: int | None = None,
        max_transitions: int | None = None,
        advance_prefix: bool = False,
        _prepared_bindings: PreparedBindings | None = None,
    ) -> NativeAnalysisResult:
        if not isinstance(inspect_accesses, bool):
            raise TypeError("native racecheck inspect_accesses must be a bool")
        if not isinstance(advance_prefix, bool):
            raise TypeError("native racecheck advance_prefix must be a bool")
        return self._run_native_racecheck_phase(
            module,
            inputs,
            phase_index=phase_index,
            subset=subset,
            inspect_accesses=inspect_accesses,
            max_polls=max_polls,
            max_transitions=max_transitions,
            advance_prefix=advance_prefix,
            prepared_bindings=_prepared_bindings,
        )

    def _run_native_synccheck_phase(
        self,
        module: CompiledModule,
        inputs: dict[str, Any],
        *,
        coverage_bounds: CoverageBounds,
        resource_limits: ResourceLimits,
        phase_index: int,
        subset: ExecutionSubset | None,
        assumptions: ExecutionAssumptions | None,
        max_polls: int | None,
        max_transitions: int | None,
        advance_prefix: bool,
        prepared_bindings: PreparedBindings | None,
    ) -> NativeAnalysisResult:
        if not isinstance(coverage_bounds, CoverageBounds):
            raise TypeError("native synccheck coverage_bounds must be CoverageBounds")
        if not isinstance(resource_limits, ResourceLimits):
            raise TypeError("native synccheck resource_limits must be ResourceLimits")
        if isinstance(phase_index, bool) or not isinstance(phase_index, int):
            raise TypeError("native synccheck phase_index must be an integer")
        if phase_index < 0 or phase_index >= len(module.spec.kernels):
            raise ValueError(
                f"native synccheck phase {phase_index} is outside [0, {len(module.spec.kernels)})"
            )
        for name, value in (("max_polls", max_polls), ("max_transitions", max_transitions)):
            if value is not None:
                _positive_policy_integer(f"analysis {name}", value)
        subset_payload = (
            None
            if subset is None
            else _normalized_execution_subset_payload(subset, field="native analysis subset")
        )
        execution = self._prepare_native_analysis_execution(
            module,
            inputs,
            assumptions=assumptions,
            prepared_bindings=prepared_bindings,
        )
        native = module.load()
        if advance_prefix:
            execution = self._advance_analysis_prefix(
                module,
                execution,
                phase_index=phase_index,
            )
        method_name = "_native_synccheck_phase"
        method = getattr(native, method_name, None)
        if method is None:
            raise NumSimExecutionError(
                f"NumSim artifact does not expose {method_name}; rebuild the generated module"
            )
        try:
            with _blas_thread_context():
                payload = method(
                    inputs=execution.bindings.to_payload(borrow_readonly=True),
                    phase_index=phase_index,
                    subset=subset_payload,
                    max_workers=self._max_workers,
                    **coverage_bounds.to_payload(),
                    **resource_limits.to_payload(),
                    max_polls=max_polls,
                    max_transitions=max_transitions,
                    native_loop_iteration_budget=self._execution_policy.native_loop_iteration_budget,
                    native_loop_reschedule_quantum=self._execution_policy.native_loop_reschedule_quantum,
                )
        except Exception as error:  # PyO3 exception boundary
            raise NumSimExecutionError(
                f"native synccheck artifact execution failed: {error}"
            ) from error
        if not isinstance(payload, dict):
            raise NumSimExecutionError(
                f"native synccheck artifact returned {type(payload).__name__}, expected dict"
            )
        verdict = payload.get("verdict")
        if verdict not in {"clean", "review", "incomplete", "error"}:
            raise NumSimExecutionError(
                f"native synccheck artifact returned invalid verdict {verdict!r}"
            )
        return NativeAnalysisResult(checker="synccheck", payload=payload)

    def _run_native_racecheck_phase(
        self,
        module: CompiledModule,
        inputs: dict[str, Any],
        *,
        phase_index: int,
        subset: ExecutionSubset | None,
        inspect_accesses: bool,
        max_polls: int | None,
        max_transitions: int | None,
        advance_prefix: bool,
        prepared_bindings: PreparedBindings | None,
    ) -> NativeAnalysisResult:
        if isinstance(phase_index, bool) or not isinstance(phase_index, int):
            raise TypeError("native racecheck phase_index must be an integer")
        if phase_index < 0 or phase_index >= len(module.spec.kernels):
            raise ValueError(
                f"native racecheck phase {phase_index} is outside [0, {len(module.spec.kernels)})"
            )
        for name, value in (("max_polls", max_polls), ("max_transitions", max_transitions)):
            if value is not None:
                _positive_policy_integer(f"racecheck {name}", value)
        subset_payload = (
            None
            if subset is None
            else _normalized_execution_subset_payload(subset, field="native racecheck subset")
        )
        # Racecheck validates one isolated launch. A grid dependency fulfilled
        # by a producer outside that launch adds no intra-launch memory edge;
        # the common NumSim launch-boundary policy discharges it automatically.
        execution = self._prepare_native_analysis_execution(
            module,
            inputs,
            assumptions=None,
            prepared_bindings=prepared_bindings,
        )
        native = module.load()
        if advance_prefix:
            execution = self._advance_analysis_prefix(
                module,
                execution,
                phase_index=phase_index,
            )
        method_name = "_native_racecheck_phase"
        method = getattr(native, method_name, None)
        if method is None:
            raise NumSimExecutionError(
                f"NumSim artifact does not expose {method_name}; rebuild the generated module"
            )
        try:
            with _blas_thread_context():
                payload = method(
                    inputs=execution.bindings.to_payload(borrow_readonly=True),
                    phase_index=phase_index,
                    subset=subset_payload,
                    inspect_accesses=inspect_accesses,
                    max_workers=self._max_workers,
                    max_polls=max_polls,
                    max_transitions=max_transitions,
                    native_loop_iteration_budget=self._execution_policy.native_loop_iteration_budget,
                    native_loop_reschedule_quantum=self._execution_policy.native_loop_reschedule_quantum,
                )
        except Exception as error:  # PyO3 exception boundary
            raise NumSimExecutionError(
                f"native racecheck artifact execution failed: {error}"
            ) from error
        if not isinstance(payload, dict):
            raise NumSimExecutionError(
                f"native racecheck artifact returned {type(payload).__name__}, expected dict"
            )
        verdict = payload.get("verdict")
        if verdict not in {"clean", "review", "incomplete", "error"}:
            raise NumSimExecutionError(
                f"native racecheck artifact returned invalid verdict {verdict!r}"
            )
        return NativeAnalysisResult(checker="racecheck", payload=payload)

    def _advance_analysis_prefix(
        self,
        module: CompiledModule,
        execution: _PreparedExecution,
        *,
        phase_index: int,
    ) -> _PreparedExecution:
        """Run earlier launches so analysis observes the real phase input state."""

        if phase_index == 0:
            return execution
        # The prefix launches write their outputs: run them on owned copies so
        # the caller's arrays stay untouched.
        execution = replace(execution, bindings=execution.bindings.freeze())
        if module.source is None:
            raise NumSimExecutionError(
                "NumSim analysis module cannot rebuild its numerical prefix artifact"
            )
        prefix_module = transpile(
            module.source,
            cache_dir=module.cache_dir,
            _default_generated_opt_level=module._generated_opt_level,
        )
        if prefix_module.spec.to_manifest(include_source_spans=False) != module.spec.to_manifest(
            include_source_spans=False
        ):
            raise NumSimExecutionError(
                "NumSim numerical prefix artifact does not match the analysis module"
            )
        native = prefix_module.load()
        method = getattr(native, "_advance_phase", None)
        if method is None:
            raise NumSimExecutionError(
                "NumSim artifact does not expose _advance_phase; rebuild the generated module"
            )
        try:
            with _blas_thread_context():
                payload = method(
                    inputs=execution.bindings.to_payload(returned_names=set()),
                    phase_index=phase_index - 1,
                    subset=None,
                    max_workers=self._max_workers,
                    native_loop_iteration_budget=self._execution_policy.native_loop_iteration_budget,
                    native_loop_reschedule_quantum=self._execution_policy.native_loop_reschedule_quantum,
                )
        except Exception as error:  # PyO3 exception boundary
            raise NumSimExecutionError(
                f"NumSim prefix execution before analysis phase {phase_index} failed: {error}"
            ) from error
        if not isinstance(payload, dict):
            raise NumSimExecutionError(
                f"NumSim prefix execution returned {type(payload).__name__}, expected dict"
            )
        states = payload.get("allocation_state")
        return replace(
            execution,
            bindings=execution.bindings.with_allocation_state(states),
        )

    def _prepare_execution(
        self,
        module: CompiledModule,
        inputs: dict[str, Any],
        *,
        outputs: Iterable[str] | Mapping[str, str] | None,
        assumptions: ExecutionAssumptions | None = None,
    ) -> _PreparedExecution:
        if isinstance(inputs, (list, tuple)):
            return self._prepare_rank_execution(
                module, inputs, outputs=outputs, assumptions=assumptions
            )
        contract = _host_abi(module)
        canonical_inputs, aliases, ambiguous_aliases = _canonicalize_buffer_names(contract, inputs)
        output_names, external_names = _resolve_output_names(
            contract, canonical_inputs, aliases, ambiguous_aliases, outputs
        )
        prepared = prepare_bindings(
            canonical_inputs,
            expected_scalar_dtypes=contract.scalar_dtypes,
            expected_buffer_dtypes=contract.buffer_dtypes,
            expected_tensor_map_names=contract.bound_tensor_map_names(canonical_inputs),
        )
        return _PreparedExecution(
            bindings=prepared,
            output_names=frozenset(output_names),
            external_names=external_names,
            assumptions=_execution_assumptions_payload(module.spec, assumptions),
        )

    def _prepare_rank_execution(
        self,
        module: CompiledModule,
        inputs: list[dict[str, Any]] | tuple[dict[str, Any], ...],
        *,
        outputs: Iterable[str] | Mapping[str, str] | None,
        assumptions: ExecutionAssumptions | None,
    ) -> _PreparedExecution:
        """One launch per rank of the same kernel, sharing one physical memory.

        Outputs are keyed ``rank_binding_name(name, rank)``; parameters bound
        to a :class:`MulticastWindow` have no output of their own.
        """

        contract = _host_abi(module)
        if not inputs or not all(isinstance(rank_inputs, dict) for rank_inputs in inputs):
            raise NumSimExecutionError("multi-rank NumSim inputs must be a non-empty list of dicts")
        canonical_ranks = []
        for rank_inputs in inputs:
            canonical, aliases, ambiguous_aliases = _canonicalize_buffer_names(
                contract, rank_inputs
            )
            canonical_ranks.append(canonical)
        output_names, external_names = _resolve_output_names(
            contract, canonical_ranks[0], aliases, ambiguous_aliases, outputs
        )
        prepared = prepare_rank_bindings(
            canonical_ranks,
            expected_scalar_dtypes=contract.scalar_dtypes,
            expected_buffer_dtypes=contract.buffer_dtypes,
            expected_tensor_map_names=[
                contract.bound_tensor_map_names(canonical) for canonical in canonical_ranks
            ],
        )
        rank_outputs: set[str] = set()
        rank_external_names: dict[str, str] = {}
        for rank, canonical in enumerate(canonical_ranks):
            for name in output_names:
                if name not in canonical or isinstance(canonical[name], MulticastWindow):
                    continue
                flat = rank_binding_name(name, rank)
                rank_outputs.add(flat)
                rank_external_names[flat] = rank_binding_name(external_names[name], rank)
        return _PreparedExecution(
            bindings=prepared,
            output_names=frozenset(rank_outputs),
            external_names=rank_external_names,
            assumptions=_execution_assumptions_payload(module.spec, assumptions),
        )

    def _prepare_native_analysis_execution(
        self,
        module: CompiledModule,
        inputs: dict[str, Any],
        *,
        assumptions: ExecutionAssumptions | None,
        prepared_bindings: PreparedBindings | None,
    ) -> _PreparedExecution:
        if prepared_bindings is None:
            # The analysis borrows the host arrays read-only (see
            # `PreparedBindings.to_payload`); only a numerical prefix run
            # needs its own copy of them.
            return self._prepare_execution(
                module,
                inputs,
                outputs=(),
                assumptions=assumptions,
            )
        contract = _host_abi(module)
        if prepared_bindings.rank_names:
            canonical_names = {
                rank_binding_name(name, rank)
                for rank, rank_inputs in enumerate(inputs)
                for name in _canonicalize_buffer_names(contract, rank_inputs)[0]
            }
        else:
            canonical_names = set(_canonicalize_buffer_names(contract, inputs)[0])
        prepared_names = prepared_bindings.buffers.keys() | prepared_bindings.scalars.keys()
        if prepared_names != canonical_names:
            raise NumSimExecutionError(
                "precomputed native-analysis bindings do not match the supplied inputs"
            )
        prepared_bindings._require_stable_host_views()
        return _PreparedExecution(
            bindings=prepared_bindings,
            output_names=frozenset(),
            external_names={},
            assumptions=_execution_assumptions_payload(module.spec, assumptions),
        )

    def _execute_prepared(
        self,
        module: CompiledModule,
        execution: _PreparedExecution,
        *,
        subset: ExecutionSubsetSelection | None,
    ) -> NumSimResult:
        prepared = execution.bindings
        output_names = set(execution.output_names)
        external_names = execution.external_names
        native = module.load()
        metadata = _native_metadata_for_execution(module, native)
        for name, expected in abi_metadata().items():
            actual = metadata.get(name)
            if actual != expected:
                raise NumSimExecutionError(
                    f"NumSim artifact {name} does not match the Python ABI: "
                    f"artifact={actual!r}, python={expected!r}"
                )
        subset_payload = _execution_subset_payload(subset, kernel_count=len(module.spec.kernels))
        try:
            with _blas_thread_context():
                payload = native.run(
                    prepared.to_payload(),
                    subset_payload,
                    self._max_workers,
                    self._execution_policy.native_loop_iteration_budget,
                    self._execution_policy.native_loop_reschedule_quantum,
                )
        except Exception as error:  # PyO3 exception boundary
            raise NumSimExecutionError(f"NumSim artifact execution failed: {error}") from error
        if not isinstance(payload, dict):
            raise NumSimExecutionError(
                f"NumSim artifact returned {type(payload).__name__}, expected dict"
            )
        allocation_bytes = payload.get("allocation_bytes")
        if not isinstance(allocation_bytes, list):
            raise NumSimExecutionError("NumSim artifact did not return physical allocation bytes")
        logical_outputs = prepared.apply_allocation_bytes(
            allocation_bytes, output_names=output_names
        )
        remapped_outputs = {
            external_names.get(name, name): value for name, value in logical_outputs.items()
        }
        return NumSimResult(
            outputs=remapped_outputs,
            diagnostics=list(payload.get("diagnostics", [])),
            stats=dict(payload.get("stats", {})),
        )


def _host_abi(module: CompiledModule) -> HostAbiContract:
    try:
        return build_host_abi(module.spec)
    except HostAbiError as error:
        raise NumSimExecutionError(f"NumSim module has an invalid host ABI: {error}") from error


def _validate_input_binding(slot: HostBindingSlot, name: str, value: Any) -> None:
    if isinstance(value, (MulticastWindow, SymmetricBuffer)) and slot.kind in {"buffer", "pointer"}:
        return
    if slot.kind in {"buffer", "pointer", "tensor_map"}:
        if not isinstance(value, np.ndarray):
            raise NumSimExecutionError(
                f"NumSim input {name!r} binds {slot.kind} slot {slot.canonical_name!r} "
                f"with {type(value).__name__}, expected a NumPy array"
            )
        return
    if isinstance(value, np.ndarray):
        raise NumSimExecutionError(
            f"NumSim input {name!r} binds scalar slot {slot.canonical_name!r} "
            f"with {type(value).__name__}"
        )


def _canonicalize_buffer_names(
    contract: HostAbiContract, inputs: dict[str, Any], *, require_all: bool = True
) -> tuple[dict[str, Any], dict[str, str], dict[str, tuple[str, ...]]]:
    aliases = contract.unique_aliases
    ambiguous_aliases = contract.ambiguous_aliases
    canonical: dict[str, Any] = {}
    selected_names: dict[str, list[str]] = {}
    for name, value in inputs.items():
        if not isinstance(name, str):
            raise TypeError("NumSim input binding names must be strings")
        if name in ambiguous_aliases:
            raise NumSimExecutionError(
                f"NumSim input alias {name!r} is ambiguous across kernels; it may identify "
                f"any of {list(ambiguous_aliases[name])}"
            )
        logical_name = aliases.get(name)
        if logical_name is None:
            expected = list(contract.known_aliases)
            suffix = "" if len(expected) <= 20 else f" (and {len(expected) - 20} more)"
            raise NumSimExecutionError(
                f"NumSim input binding {name!r} is unknown; expected one of {expected[:20]}{suffix}"
            )
        if logical_name in canonical:
            used = [*selected_names[logical_name], name]
            raise NumSimExecutionError(
                f"NumSim binding {logical_name!r} was provided through multiple aliases: {used}"
            )
        _validate_input_binding(contract.slot(logical_name), name, value)
        canonical[logical_name] = value
        selected_names.setdefault(logical_name, []).append(name)
    required = {
        slot.canonical_name
        for slot in contract.slots
        if slot.canonical_name not in contract.implicit_tensor_map_names
    }
    missing = sorted(required - canonical.keys())
    if require_all and missing:
        raise NumSimExecutionError(f"NumSim inputs are missing required bindings: {missing}")
    return canonical, aliases, ambiguous_aliases


def _resolve_output_names(
    contract: HostAbiContract,
    canonical_inputs: dict[str, Any],
    aliases: dict[str, str],
    ambiguous_aliases: dict[str, tuple[str, ...]],
    outputs: Iterable[str] | Mapping[str, str] | None,
) -> tuple[set[str], dict[str, str]]:
    if outputs is None:
        bound = set(canonical_inputs) & contract.output_binding_names
        return bound, {name: name for name in bound}

    if isinstance(outputs, str):
        raise TypeError("NumSim outputs must be an iterable of names, not a string")
    selections = (
        outputs.items() if isinstance(outputs, Mapping) else ((name, name) for name in outputs)
    )
    logical_names: set[str] = set()
    external_names: dict[str, str] = {}
    for external_name, selected_name in selections:
        if not isinstance(external_name, str) or not isinstance(selected_name, str):
            raise TypeError("NumSim output names and selectors must be strings")
        if selected_name in ambiguous_aliases:
            raise NumSimExecutionError(
                f"NumSim output alias {selected_name!r} is ambiguous across kernels; it may "
                f"identify any of {list(ambiguous_aliases[selected_name])}"
            )
        logical_name = aliases.get(selected_name, selected_name)
        if (
            logical_name not in canonical_inputs
            or logical_name not in contract.output_binding_names
        ):
            raise NumSimExecutionError(
                f"NumSim output selector {selected_name!r} does not identify a bound kernel buffer"
            )
        external_alias = aliases.get(external_name)
        if external_alias is not None and external_alias != logical_name:
            raise NumSimExecutionError(
                f"NumSim output name {external_name!r} collides with kernel buffer "
                f"{external_alias!r}"
            )
        if logical_name in external_names:
            raise NumSimExecutionError(
                f"NumSim buffer {logical_name!r} is exposed as more than one output"
            )
        logical_names.add(logical_name)
        external_names[logical_name] = external_name
    return logical_names, external_names


@cache_semantic_ir_serializations()
def transpile(
    func: Any,
    *,
    cache_dir: str | Path | None = None,
    _default_generated_opt_level: int = 3,
    _analysis_capable: bool = False,
    _analysis_checker: Literal["synccheck", "racecheck"] | None = None,
) -> CompiledModule:
    """Verify, emit, build, and cache a native Rust NumSim artifact."""

    from .transpiler.build import (
        build_artifact,
        load_cached_generated_artifact,
        prepare_generated_artifact,
    )
    from .transpiler.host_prelude import normalize_transpile_source

    if _analysis_checker not in {None, "synccheck", "racecheck"}:
        raise ValueError(f"unknown native analysis checker: {_analysis_checker!r}")
    analysis_capable = _analysis_capable or _analysis_checker is not None
    frozen_source = normalize_transpile_source(
        tuple(func) if isinstance(func, (list, tuple)) else func
    )
    prepared = prepare_generated_artifact(
        frozen_source,
        cache_dir=cache_dir,
        analysis_capable=analysis_capable,
        analysis_checker=_analysis_checker,
        default_opt_level=_default_generated_opt_level,
    )
    artifact = load_cached_generated_artifact(prepared)
    if artifact is None:
        from .transpiler.compile_cache import compile_module_cached

        spec, source_template = compile_module_cached(
            frozen_source,
            analysis_capable=analysis_capable,
            analysis_checker=_analysis_checker,
        )
        artifact = build_artifact(
            spec,
            source_template,
            cache_dir=cache_dir,
            prepared=prepared,
        )
    elif artifact.spec is None:
        raise NumSimBuildError("generated artifact cache is missing its module spec")
    else:
        try:
            spec = attach_source_nodes(
                artifact.spec, frozen_source, _render_script=False, _cache_key=prepared.key
            )
        except ValueError as error:
            raise NumSimBuildError(
                f"generated artifact cache module spec does not match its source: {error}"
            ) from error
    resolved_cache = None if cache_dir is None else Path(cache_dir).expanduser().resolve()
    return CompiledModule(
        spec=spec,
        artifact=artifact,
        source=frozen_source,
        cache_dir=resolved_cache,
        _generated_opt_level=prepared.build_config.release_opt_level,
        _analysis_capable=analysis_capable,
        _analysis_checker=_analysis_checker,
    )


def dump_rust(module: CompiledModule) -> str:
    """Return the exact Rust source compiled for an artifact."""
    return module.rust_source


def dump_semantic_manifest(module: CompiledModule) -> dict[str, Any]:
    """Return the serializable compile-time TIRx coverage/source manifest."""
    return module.spec.to_manifest()


def compare(
    result: NumSimResult,
    expected: dict[str, Any],
    *,
    tolerances: dict[str, ComparisonSpec] | None = None,
) -> NumSimReport:
    if not expected:
        raise NumSimExecutionError("NumSim expected outputs must not be empty")
    tolerances = {} if tolerances is None else tolerances
    unknown_tolerances = tolerances.keys() - expected.keys()
    if unknown_tolerances:
        raise NumSimExecutionError(
            f"NumSim comparison specs refer to unknown expected outputs: "
            f"{sorted(unknown_tolerances)}"
        )
    mismatches: list[Mismatch] = []
    for name, reference in expected.items():
        if name not in result.outputs:
            mismatches.append(Mismatch(name, (), "<missing>", "<present>"))
            continue
        actual_array = _decode_comparison_array(
            np.asarray(result.outputs[name]), tolerances.get(name, ComparisonSpec()).actual_encoding
        )
        expected_array = np.asarray(reference)
        explicit_tolerance = name in tolerances
        spec = tolerances.get(name, ComparisonSpec())
        regions = spec.regions or (
            ComparisonRegion(
                tuple(slice(None) for _ in actual_array.shape),
                tuple(slice(None) for _ in expected_array.shape),
            ),
        )
        for region in regions:
            actual_selector = _normalize_comparison_selector(
                region.actual, actual_array.ndim, field=f"{name}.actual"
            )
            expected_selector = _normalize_comparison_selector(
                region.expected if region.expected is not None else region.actual,
                expected_array.ndim,
                field=f"{name}.expected",
            )
            actual_view = np.asarray(actual_array[actual_selector])
            expected_view = np.asarray(expected_array[expected_selector])
            if actual_view.size == 0 or expected_view.size == 0:
                raise NumSimExecutionError(
                    f"NumSim comparison region for {name!r} selects zero elements"
                )
            if actual_view.shape != expected_view.shape:
                mismatches.append(
                    Mismatch(
                        name,
                        _comparison_index(actual_selector, (), actual_array.shape),
                        actual_view.shape,
                        expected_view.shape,
                    )
                )
                break
            exact_dtypes = {"b", "i", "u"}
            actual_numeric = _comparison_numeric_view(actual_view)
            expected_numeric = _comparison_numeric_view(expected_view)
            if (
                not explicit_tolerance
                and actual_view.dtype.kind in exact_dtypes
                and expected_view.dtype.kind in exact_dtypes
            ):
                close = np.equal(actual_view, expected_view)
            elif actual_numeric is not None and expected_numeric is not None:
                close = np.isclose(
                    actual_numeric,
                    expected_numeric,
                    rtol=spec.rtol,
                    atol=spec.atol,
                    equal_nan=spec.equal_nan,
                )
            else:
                close = np.equal(actual_view, expected_view)
            if np.all(close):
                continue
            first = tuple(int(value) for value in np.argwhere(~close)[0])
            mismatches.append(
                Mismatch(
                    name,
                    _comparison_index(actual_selector, first, actual_array.shape),
                    actual_view[first].item(),
                    expected_view[first].item(),
                )
            )
            break
    return NumSimReport(ok=not mismatches, mismatches=mismatches, diagnostics=result.diagnostics)


def _comparison_numeric_view(array: np.ndarray) -> np.ndarray | None:
    if array.dtype.kind in "biufc":
        return array
    if array.dtype.name == "bfloat16":
        return array.astype(np.float32)
    return None


def _decode_comparison_array(array: np.ndarray, encoding: str | None) -> np.ndarray:
    if encoding is None:
        return array
    if encoding == "bfloat16":
        if array.dtype.kind not in "iu":
            raise NumSimExecutionError(
                f"bfloat16 comparison requires an integer backing, got {array.dtype}"
            )
        if array.dtype.itemsize == 2:
            encoded = array.astype(np.uint16, copy=False)
        elif array.dtype.itemsize == 1:
            if array.ndim != 1 or array.size % 2:
                raise NumSimExecutionError(
                    "bfloat16 byte-carrier comparison requires a one-dimensional even byte count"
                )
            encoded = np.ascontiguousarray(array).view(np.uint8).view(np.uint16)
        else:
            raise NumSimExecutionError(
                "bfloat16 comparison requires a 16-bit integer or 8-bit byte-carrier "
                f"backing, got {array.dtype}"
            )
        bits = encoded.astype(np.uint32) << np.uint32(16)
        return bits.view(np.float32)
    raise AssertionError(f"unvalidated NumSim comparison encoding {encoding!r}")


def _normalize_comparison_selector(
    selector: tuple[int | slice, ...], rank: int, *, field: str
) -> tuple[int | slice, ...]:
    if len(selector) > rank:
        raise NumSimExecutionError(
            f"NumSim comparison selector {field} has rank {len(selector)}, exceeding array rank {rank}"
        )
    normalized: list[int | slice] = []
    for value in selector:
        if isinstance(value, bool) or not isinstance(value, (int, slice)):
            raise NumSimExecutionError(
                f"NumSim comparison selector {field} contains unsupported index {value!r}"
            )
        normalized.append(value)
    normalized.extend(slice(None) for _ in range(rank - len(normalized)))
    return tuple(normalized)


def _comparison_index(
    selector: tuple[int | slice, ...], local: tuple[int, ...], shape: tuple[int, ...]
) -> tuple[int, ...]:
    result: list[int] = []
    local_axis = 0
    for axis, value in enumerate(selector):
        if isinstance(value, int):
            result.append(value + shape[axis] if value < 0 else value)
            continue
        start, stop, step = value.indices(shape[axis])
        if local_axis >= len(local):
            result.append(start)
        else:
            coordinate = start + local[local_axis] * step
            if coordinate < 0 or coordinate >= shape[axis] or coordinate == stop:
                raise NumSimExecutionError("NumSim comparison mismatch index escaped its region")
            result.append(coordinate)
        local_axis += 1
    return tuple(result)


def run_case(case: NumSimCase, *, engine: Engine | None = None) -> NumSimReport:
    engine = engine or Engine()
    module = transpile(case.kernel)
    execution = engine._prepare_execution(
        module, case.args, outputs=case.outputs, assumptions=case.assumptions
    )
    execution = replace(execution, bindings=execution.bindings.freeze())
    try:
        expected = copy.deepcopy(case.reference())
    finally:
        execution.bindings.restore_host_buffers()
    if not expected:
        raise NumSimExecutionError("NumSim expected outputs must not be empty")
    if not set(expected) <= execution.selected_output_names:
        raise NumSimExecutionError(
            "NumSim reference outputs must name selected kernel outputs: "
            f"selected={sorted(execution.selected_output_names)}, "
            f"reference={sorted(expected)}"
        )
    result = engine._execute_prepared(module, execution, subset=case.subset)
    return compare(result, expected, tolerances=case.comparisons)
