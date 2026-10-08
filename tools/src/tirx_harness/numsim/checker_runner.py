"""Shared binding and metadata contract for native analysis tools."""

from __future__ import annotations

import copy
import hashlib
import json
import threading
from concurrent.futures import Future
from dataclasses import dataclass, field as dataclass_field, replace
from typing import Any, Literal

import numpy as np
from tvm import tirx
from tvm.ir.type import PointerType

from .bindings import PreparedBindings, prepare_bindings
from .errors import NumSimExecutionError, UnsupportedTIRxError
from .host_abi import HostAbiContract, HostBindingSlot, build_host_abi
from .transpiler.frontend import _extract_topology
from .transpiler.source_map import (
    deserialize_source_span,
    flatten_source_span,
    serialize_source_span,
    source_span_from_node,
)

NativeChecker = Literal["synccheck", "racecheck"]
_NATIVE_ANALYSIS_DEFAULT_GENERATED_OPT_LEVEL = 3


@dataclass(frozen=True)
class NativeBindingPreparation:
    """One frozen, canonical host snapshot or an actionable input gap."""

    inputs: dict[str, Any]
    canonical_inputs: dict[str, Any]
    execution_inputs: dict[str, Any]
    input_digest: str | None
    scalar_bindings: dict[str, Any]
    missing_bindings: tuple[str, ...]
    prepared_bindings: PreparedBindings | None = dataclass_field(
        default=None, repr=False, compare=False
    )
    _identity_future: Future[tuple[str, dict[str, Any]]] | None = dataclass_field(
        default=None, repr=False, compare=False
    )

    @property
    def complete(self) -> bool:
        return not self.missing_bindings

    def resolve_identity(self) -> NativeBindingPreparation:
        if self._identity_future is None:
            return self
        input_digest, scalar_bindings = self._identity_future.result()
        return replace(
            self,
            input_digest=input_digest,
            scalar_bindings=scalar_bindings,
            _identity_future=None,
        )


@dataclass(frozen=True)
class NativeCheckerRun:
    module: Any | None
    bindings: NativeBindingPreparation | None
    payload: dict[str, Any]


def default_coverage_bounds():
    """Resolve the shared checker coverage default owned by ``api``."""

    from .api import default_coverage_bounds as shared

    return shared()


def default_resource_limits():
    """Resolve the shared checker resource default owned by ``api``."""

    from .api import default_resource_limits as shared

    return shared()


def _slot_for_alias(contract: HostAbiContract, name: str) -> HostBindingSlot | None:
    targets = contract.ambiguous_aliases.get(name)
    if targets is not None:
        raise NumSimExecutionError(
            f"native analysis input alias {name!r} is ambiguous across {list(targets)}"
        )
    canonical = contract.unique_aliases.get(name)
    return None if canonical is None else contract.slot(canonical)


def _validate_explicit_binding(slot: HostBindingSlot, name: str, value: Any) -> None:
    if slot.kind in {"buffer", "pointer", "tensor_map"}:
        if not isinstance(value, np.ndarray):
            raise NumSimExecutionError(f"native analysis input {name!r} must be a NumPy array")
        return
    if isinstance(value, np.ndarray):
        raise NumSimExecutionError(f"native analysis scalar input {name!r} has a buffer value")


def _canonicalize_inputs(contract: HostAbiContract, inputs: dict[str, Any]) -> dict[str, Any]:
    canonical: dict[str, Any] = {}
    selected_names: dict[str, str] = {}
    for name, value in inputs.items():
        if not isinstance(name, str):
            raise TypeError("native analysis input names must be strings")
        slot = _slot_for_alias(contract, name)
        if slot is None:
            raise NumSimExecutionError(
                f"native analysis input {name!r} is unknown; expected one of "
                f"{list(contract.known_aliases)}"
            )
        previous = selected_names.get(slot.canonical_name)
        if previous is not None:
            raise NumSimExecutionError(
                f"native analysis binding {slot.canonical_name!r} was supplied through "
                f"both {previous!r} and {name!r}"
            )
        _validate_explicit_binding(slot, name, value)
        canonical[slot.canonical_name] = value
        selected_names[slot.canonical_name] = name
    return canonical


def prepare_native_bindings(
    module: Any,
    *,
    inputs: dict[str, Any] | None,
    evidence_inputs: dict[str, Any] | None = None,
    supplemental_scalar_dtypes: dict[str, str] | None = None,
    _defer_identity: bool = False,
) -> NativeBindingPreparation:
    """Freeze the exact concrete inputs for one native checker execution."""

    contract = build_host_abi(module.spec)
    if inputs is not None and not isinstance(inputs, dict):
        raise TypeError("native analysis inputs must be a dict or None")
    public_inputs = dict(inputs or {})
    public_evidence_inputs = dict(public_inputs if evidence_inputs is None else evidence_inputs)
    supplemental_scalar_dtypes = dict(supplemental_scalar_dtypes or {})

    canonical = _canonicalize_inputs(contract, public_inputs)
    supplemental_inputs = {
        name: public_evidence_inputs[name]
        for name in supplemental_scalar_dtypes
        if name in public_evidence_inputs
    }
    missing_supplemental = tuple(
        sorted(supplemental_scalar_dtypes.keys() - supplemental_inputs.keys())
    )
    if missing_supplemental:
        raise NumSimExecutionError(
            "native analysis launch scalar bindings disappeared before execution: "
            f"{list(missing_supplemental)}"
        )
    collision = tuple(sorted(canonical.keys() & supplemental_inputs.keys()))
    if collision:
        raise NumSimExecutionError(
            "native analysis launch scalar bindings collide with artifact bindings: "
            f"{list(collision)}"
        )
    supplemental_prepared = prepare_bindings(
        supplemental_inputs,
        expected_scalar_dtypes=supplemental_scalar_dtypes,
    )
    evidence_canonical = {**canonical, **supplemental_inputs}
    required = {
        slot.canonical_name
        for slot in contract.slots
        if slot.canonical_name not in contract.implicit_tensor_map_names
    }
    missing = tuple(sorted(required - canonical.keys()))
    if missing:
        return NativeBindingPreparation(
            inputs=public_evidence_inputs,
            canonical_inputs=evidence_canonical,
            execution_inputs=canonical,
            input_digest=None,
            scalar_bindings=copy.deepcopy(
                supplemental_prepared.identity_payload()["scalars"]
            ),
            missing_bindings=missing,
            prepared_bindings=None,
            _identity_future=None,
        )

    prepared = prepare_bindings(
        canonical,
        expected_scalar_dtypes=contract.scalar_dtypes,
        expected_buffer_dtypes=contract.buffer_dtypes,
        expected_tensor_map_names=contract.bound_tensor_map_names(canonical),
    ).freeze()
    identity_prepared = replace(
        prepared,
        scalars={**prepared.scalars, **supplemental_prepared.scalars},
    )
    if _defer_identity:
        identity_future = _deferred_native_binding_identity(identity_prepared)
        digest = None
        scalar_bindings = {}
    else:
        digest, scalar_bindings = _native_binding_identity(identity_prepared)
        identity_future = None
    return NativeBindingPreparation(
        inputs=public_evidence_inputs,
        canonical_inputs=evidence_canonical,
        execution_inputs=canonical,
        input_digest=digest,
        scalar_bindings=scalar_bindings,
        missing_bindings=(),
        prepared_bindings=prepared,
        _identity_future=identity_future,
    )


def _concretize_checker_launch(
    func: Any,
    inputs: dict[str, Any] | None,
) -> tuple[Any, dict[str, Any] | None, dict[str, str]]:
    """Specialize only scalar inputs required to determine the concrete launch."""

    if inputs is None or not isinstance(inputs, dict) or not hasattr(func, "params"):
        return func, inputs, {}

    scalar_parameters = {
        str(parameter.name): parameter
        for parameter in func.params
        if not tirx.is_buffer_var(parameter) and not isinstance(parameter.ty, PointerType)
    }
    provided: dict[Any, int | float | bool] = {}
    for name, parameter in scalar_parameters.items():
        if name not in inputs:
            continue
        value = inputs[name]
        if isinstance(value, np.ndarray):
            raise NumSimExecutionError(
                f"native analysis scalar input {name!r} has a buffer value"
            )
        prepared = prepare_bindings(
            {name: value},
            expected_scalar_dtypes={name: str(parameter.ty.dtype)},
        )
        try:
            provided[parameter] = prepared.scalars[name].value
        except KeyError as error:
            raise NumSimExecutionError(
                f"native analysis scalar input {name!r} is not a scalar value"
            ) from error
    if not provided:
        return func, inputs, {}

    concrete_topology = _extract_topology(func, provided)
    required = dict(provided)
    for parameter in provided:
        candidate = {
            key: value for key, value in required.items() if not key.same_as(parameter)
        }
        try:
            candidate_topology = _extract_topology(func, candidate)
        except UnsupportedTIRxError:
            continue
        if candidate_topology == concrete_topology:
            required = candidate

    if not required:
        return func, inputs, {}

    specialized = func.specialize(required)
    execution_inputs = dict(inputs)
    supplemental_scalar_dtypes: dict[str, str] = {}
    for parameter in required:
        name = str(parameter.name)
        execution_inputs.pop(name)
        supplemental_scalar_dtypes[name] = str(parameter.ty.dtype)
    return specialized, execution_inputs, supplemental_scalar_dtypes


def _native_binding_identity(
    prepared: PreparedBindings,
    *,
    before_first_data_hash: Any | None = None,
) -> tuple[str, dict[str, Any]]:
    identity = prepared.identity_payload(
        _before_first_data_hash=before_first_data_hash,
    )
    digest = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return digest, copy.deepcopy(identity["scalars"])


def _deferred_native_binding_identity(
    prepared: PreparedBindings,
) -> Future[tuple[str, dict[str, Any]]]:
    future: Future[tuple[str, dict[str, Any]]] = Future()
    hash_started = threading.Event()

    def resolve() -> None:
        try:
            future.set_result(
                _native_binding_identity(
                    prepared,
                    before_first_data_hash=hash_started.set,
                )
            )
        except BaseException as error:
            future.set_exception(error)
        finally:
            hash_started.set()

    threading.Thread(
        target=resolve,
        name="racecheck-input-identity",
        daemon=True,
    ).start()
    hash_started.wait()
    return future


def _attach_native_metadata_owned(
    payload: dict[str, Any], module: Any, bindings: NativeBindingPreparation
) -> dict[str, Any]:
    """Attach metadata to a freshly produced payload whose ownership is transferred here."""

    _attach_native_source_evidence_owned(payload, module)
    metadata = native_evidence_metadata(module, bindings)
    payload["engine"] = metadata["engine"]
    payload["input"] = metadata["input"]
    return payload


def _attach_native_source_evidence_owned(payload: dict[str, Any], module: Any) -> None:
    """Attach the best available operation or kernel evidence to native diagnostics."""

    diagnostic_records = list(_native_diagnostic_records(payload))
    if payload.get("verdict") == "clean" and not diagnostic_records:
        return

    sites = {
        (kernel_index, source.op_id): {
            "source_op_id": source.op_id,
            "kind": source.kind,
            **({"op_name": source.op_name} if source.op_name is not None else {}),
            "source_text": source.text,
            "source_span": serialize_source_span(source.span),
        }
        for kernel_index, kernel in enumerate(module.spec.kernels)
        for source in kernel.source_map
    }

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            kernel_index = value.get("kernel_index")
            source_op_id = value.get("source_op_id")
            if (
                isinstance(kernel_index, int)
                and not isinstance(kernel_index, bool)
                and isinstance(source_op_id, int)
                and not isinstance(source_op_id, bool)
            ):
                source = sites.get((kernel_index, source_op_id))
                if source is not None:
                    value.setdefault("source", copy.deepcopy(source))
            for key, child in tuple(value.items()):
                if key != "source":
                    visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    for _, details in diagnostic_records:
        visit(details)

    anchors = {
        kernel_index: _kernel_source_anchor(kernel_index, kernel)
        for kernel_index, kernel in enumerate(module.spec.kernels)
    }
    if not anchors:
        raise NumSimExecutionError("native analysis cannot report diagnostics without a kernel")
    default_kernel_index = _payload_phase_index(payload)
    default_anchor = anchors.get(default_kernel_index, anchors[min(anchors)])
    payload.setdefault("source_anchor", copy.deepcopy(default_anchor))

    for field, details in diagnostic_records:
        if not _native_diagnostic_has_source_line(details):
            kernel_index = _first_kernel_index(details)
            details["source_anchor"] = copy.deepcopy(anchors.get(kernel_index, default_anchor))
        if not _native_diagnostic_has_source_line(details):
            kind = details.get("kind", field)
            raise NumSimExecutionError(
                f"native {field} diagnostic {kind!r} has no structured source line"
            )


def _serialized_source_span_has_line(value: Any) -> bool:
    try:
        return bool(flatten_source_span(deserialize_source_span(value)))
    except ValueError:
        return False


def _native_diagnostic_has_source_line(value: Any) -> bool:
    if isinstance(value, dict):
        if _serialized_source_span_has_line(value.get("source_span")) or (
            isinstance(value.get("source_text"), str) and value["source_text"].strip()
        ):
            return True
        return any(
            _native_diagnostic_has_source_line(child)
            for key, child in value.items()
            if key != "source_span"
        )
    if isinstance(value, (list, tuple)):
        return any(_native_diagnostic_has_source_line(child) for child in value)
    return False


def _kernel_source_anchor(kernel_index: int, kernel: Any) -> dict[str, Any]:
    for source in reversed(kernel.source_map):
        span = serialize_source_span(source.span)
        if _serialized_source_span_has_line(span):
            return {
                "scope": "kernel",
                "kernel_index": kernel_index,
                "source_text": f"kernel {kernel.name}",
                "source_span": span,
            }
    return {
        "scope": "kernel",
        "kernel_index": kernel_index,
        "source_text": f"kernel {kernel.name}",
        "source_span": None,
    }


def _payload_phase_index(payload: dict[str, Any]) -> int:
    phase = payload.get("phase")
    if isinstance(phase, dict):
        index = phase.get("index")
        if isinstance(index, int) and not isinstance(index, bool) and index >= 0:
            return index
    return 0


def _first_kernel_index(value: Any) -> int | None:
    if isinstance(value, dict):
        kernel_index = value.get("kernel_index")
        if (
            isinstance(kernel_index, int)
            and not isinstance(kernel_index, bool)
            and kernel_index >= 0
        ):
            return kernel_index
        for child in value.values():
            if (kernel_index := _first_kernel_index(child)) is not None:
                return kernel_index
    elif isinstance(value, (list, tuple)):
        for child in value:
            if (kernel_index := _first_kernel_index(child)) is not None:
                return kernel_index
    return None


def _native_diagnostic_records(payload: dict[str, Any]):
    has_typed_error = False
    for field, fallback_key in (
        ("findings", "finding"),
        ("advisories", "advisory"),
        ("incomplete", "reason"),
    ):
        values = payload.get(field)
        if not isinstance(values, list):
            continue
        for index, value in enumerate(values):
            if not isinstance(value, dict):
                value = {fallback_key: value}
                values[index] = value
            if field == "findings" and value.get("status") == "error":
                has_typed_error = True
            yield field, value
    execution_error = payload.get("execution_error")
    if isinstance(execution_error, dict) and not has_typed_error:
        yield "execution_error", execution_error


def _primfunc_source_anchor(func: Any, *, kernel_index: int = 0) -> dict[str, Any] | None:
    body = getattr(func, "body", None)
    if body is not None:
        span = serialize_source_span(source_span_from_node(body))
        if _serialized_source_span_has_line(span):
            name = getattr(func, "__name__", None)
            attrs = getattr(func, "attrs", None)
            if attrs is not None:
                try:
                    name = str(attrs["global_symbol"])
                except (KeyError, TypeError):
                    pass
            return {
                "scope": "kernel",
                "kernel_index": kernel_index,
                "source_text": f"kernel {name or kernel_index}",
                "source_span": span,
            }

    if isinstance(func, (list, tuple)):
        for index, candidate in enumerate(func):
            if (anchor := _primfunc_source_anchor(candidate, kernel_index=index)) is not None:
                return anchor

    functions = getattr(func, "functions", None)
    if functions is not None:
        try:
            candidates = functions.values()
        except AttributeError:
            candidates = ()
        for index, candidate in enumerate(candidates):
            if (anchor := _primfunc_source_anchor(candidate, kernel_index=index)) is not None:
                return anchor
    return None


def native_evidence_metadata(module: Any, bindings: NativeBindingPreparation) -> dict[str, Any]:
    """Return engine and concrete-input identity without changing a strict payload."""

    return {
        "engine": {
            "mode": "native",
            "artifact_key": module.cache_key,
        },
        "input": {
            "digest": bindings.input_digest,
            "bindings": sorted(bindings.canonical_inputs),
            "scalars": copy.deepcopy(bindings.scalar_bindings),
        },
    }


def native_binding_incomplete_payload(
    checker: NativeChecker, module: Any, bindings: NativeBindingPreparation
) -> dict[str, Any]:
    """Return a typed fail-closed result when concrete native inputs are absent."""

    if bindings.complete:
        raise ValueError("native binding incomplete payload requires an incomplete preparation")
    kernel = module.spec.kernels[0]
    topology = kernel.topology
    details = {
        "kind": "analysis_incomplete",
        "reason": "missing_input_bindings",
        "message": "native analysis requires complete concrete bindings before execution",
        "bindings": list(bindings.missing_bindings),
    }
    payload: dict[str, Any] = {
        "schema_version": 3,
        "phase": {
            "index": 0,
            "name": kernel.name,
            "topology": {
                "clusters": topology.clusters,
                "ctas_per_cluster": topology.ctas_per_cluster,
                "warps_per_cta": topology.warps_per_cta,
                "warp_count": topology.warp_count,
            },
        },
        "analysis_scope": {
            "kind": "full_launch",
            "selected_warp_count": topology.warp_count,
            "total_warp_count": topology.warp_count,
        },
        "verdict": "incomplete",
        "findings": [],
        "advisories": [],
        "incomplete": [details],
        "stats": {"available": False},
        "execution_error": None,
        "coverage": {
            "status": "not_started",
            "eligible_for_clean": False,
            "termination": {"kind": "missing_input_bindings"},
        },
    }
    if checker == "synccheck":
        payload["effects"] = []
    else:
        payload.update(
            {
                "execution_model": "direct_online_vc",
                "sync": {
                    "verdict": "incomplete",
                    "findings": [],
                    "incomplete": [],
                    "effects": [],
                },
                "access_count": 0,
                "accesses_complete": False,
                "accesses": [],
            }
        )
    return _attach_native_metadata_owned(payload, module, bindings)


def _frontend_incomplete_payload(
    checker: NativeChecker,
    error: UnsupportedTIRxError,
    *,
    source_anchor: dict[str, Any] | None,
) -> dict[str, Any]:
    details: dict[str, Any] = {
        "kind": "analysis_incomplete",
        "reason": "native_frontend_unsupported",
        "message": str(error),
    }
    # Prefer the span of the node that made the kernel unsupported. The finding
    # is about that node, so nothing here needs a whole-function anchor to
    # exist first -- a kernel whose root statement carries no span (any caller
    # that recomposed it) still gets a located, structured report.
    node_span = serialize_source_span(getattr(error, "source_span", None))
    if _serialized_source_span_has_line(node_span):
        details["source_anchor"] = {
            "scope": "operation",
            "kernel_index": 0,
            "source_text": str(error),
            "source_span": node_span,
        }
    elif source_anchor is not None and _native_diagnostic_has_source_line(source_anchor):
        details["source_anchor"] = copy.deepcopy(source_anchor)
    payload: dict[str, Any] = {
        "schema_version": 3,
        "verdict": "incomplete",
        "findings": [],
        "advisories": [],
        "incomplete": [details],
        "stats": {"available": False},
        "execution_error": None,
        "coverage": {
            "status": "not_started",
            "eligible_for_clean": False,
            "termination": {"kind": "native_frontend_unsupported"},
        },
        "engine": {
            "mode": "native",
            "artifact_key": None,
        },
        "input": {"digest": None, "bindings": [], "scalars": {}},
        "source_anchor": copy.deepcopy(source_anchor),
    }
    if checker == "synccheck":
        payload["effects"] = []
    else:
        payload.update(
            {
                "execution_model": "direct_online_vc",
                "sync": {
                    "verdict": "incomplete",
                    "findings": [],
                    "incomplete": [],
                    "effects": [],
                },
                "access_count": 0,
                "accesses_complete": False,
                "accesses": [],
            }
        )
    return payload


def transpile_native_checker_artifact(
    checker: NativeChecker,
    func: Any,
    *,
    cache_dir: Any | None = None,
) -> Any:
    """Transpile with the artifact policy used by production native analysis."""

    from .api import transpile

    return transpile(
        func,
        cache_dir=cache_dir,
        _default_generated_opt_level=_NATIVE_ANALYSIS_DEFAULT_GENERATED_OPT_LEVEL,
        _analysis_checker=checker,
    )


def run_native_checker(
    checker: NativeChecker,
    func: Any,
    *,
    inputs: dict[str, Any] | None,
    coverage_bounds: Any | None = None,
    resource_limits: Any | None = None,
    subset: Any | None = None,
    cache_dir: Any | None = None,
    max_workers: int | Literal["auto"] = 8,
    max_polls: int | None = None,
    max_transitions: int | None = None,
    native_loop_iteration_budget: int = 1_000_000,
    native_loop_reschedule_quantum: int = 64,
) -> NativeCheckerRun:
    """Transpile once and run one native checker with no implicit fallback."""

    from .api import Engine

    try:
        concrete_func, execution_inputs, launch_scalar_dtypes = _concretize_checker_launch(
            func, inputs
        )
        module = transpile_native_checker_artifact(checker, concrete_func, cache_dir=cache_dir)
    except UnsupportedTIRxError as error:
        return NativeCheckerRun(
            module=None,
            bindings=None,
            payload=_frontend_incomplete_payload(
                checker, error, source_anchor=_primfunc_source_anchor(func)
            ),
        )
    if len(module.spec.kernels) != 1:
        error = UnsupportedTIRxError(
            "public native analysis currently requires exactly one kernel phase"
        )
        return NativeCheckerRun(
            module=None,
            bindings=None,
            payload=_frontend_incomplete_payload(
                checker,
                error,
                source_anchor=_kernel_source_anchor(0, module.spec.kernels[0]),
            ),
        )

    bindings = prepare_native_bindings(
        module,
        inputs=execution_inputs,
        evidence_inputs=inputs,
        supplemental_scalar_dtypes=launch_scalar_dtypes,
        _defer_identity=checker == "racecheck",
    )
    if not bindings.complete:
        return NativeCheckerRun(
            module=module,
            bindings=bindings,
            payload=native_binding_incomplete_payload(checker, module, bindings),
        )

    engine = Engine(
        max_workers=max_workers,
        native_loop_iteration_budget=native_loop_iteration_budget,
        native_loop_reschedule_quantum=native_loop_reschedule_quantum,
    )
    if checker == "racecheck":
        result = engine.run_racecheck_phase(
            module,
            bindings.execution_inputs,
            phase_index=0,
            subset=subset,
            inspect_accesses=False,
            max_polls=max_polls,
            max_transitions=max_transitions,
            _prepared_bindings=bindings.prepared_bindings,
        )
        payload = getattr(result, "payload", None)
        if not isinstance(payload, dict):
            payload = result.to_dict()
        bindings = bindings.resolve_identity()
        return NativeCheckerRun(
            module=module,
            bindings=replace(bindings, prepared_bindings=None),
            payload=_attach_native_metadata_owned(payload, module, bindings),
        )

    coverage_bounds = coverage_bounds or default_coverage_bounds()
    resource_limits = resource_limits or default_resource_limits()
    run = getattr(engine, "run_synccheck_phase", None)
    if run is None:
        raise NumSimExecutionError("native engine does not expose production synccheck execution")
    result = run(
        module,
        bindings.execution_inputs,
        coverage_bounds=coverage_bounds,
        resource_limits=resource_limits,
        phase_index=0,
        subset=subset,
        max_polls=max_polls,
        max_transitions=max_transitions,
        _prepared_bindings=bindings.prepared_bindings,
    )
    payload = getattr(result, "payload", None)
    if not isinstance(payload, dict):
        payload = result.to_dict()
    return NativeCheckerRun(
        module=module,
        bindings=replace(bindings, prepared_bindings=None),
        payload=_attach_native_metadata_owned(payload, module, bindings),
    )


__all__ = [
    "NativeBindingPreparation",
    "NativeCheckerRun",
    "default_coverage_bounds",
    "default_resource_limits",
    "native_binding_incomplete_payload",
    "native_evidence_metadata",
    "prepare_native_bindings",
    "run_native_checker",
    "transpile_native_checker_artifact",
]
