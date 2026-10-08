"""Installed, compile-time services backed by the unmodified TVM Rust extension."""

from __future__ import annotations

import json
from functools import cache, partial
from importlib.machinery import EXTENSION_SUFFIXES
from importlib.metadata import distributions
from pathlib import Path
from typing import Any


from ..errors import NumSimBuildError, UnmodeledTIRxFormError, UnsupportedTIRxError


@cache
def library_path() -> Path:
    package = Path(__file__).resolve().parent.parent
    for suffix in EXTENSION_SUFFIXES:
        candidate = package / f"_tvm_rust_ext{suffix}"
        if candidate.is_file():
            return candidate
    # Repository tests import Python sources directly after installing the
    # wheel. Locate its native component through distribution metadata, never
    # through the repository's parent or a thirdparty source path.
    for installed in distributions(name="tirx-harness"):
        for file in installed.files or ():
            if str(file).startswith("tirx_harness/numsim/_tvm_rust_ext.") and file.suffix in {".so", ".pyd"}:
                candidate = Path(installed.locate_file(file)).resolve()
                if candidate.is_file():
                    return candidate
    raise NumSimBuildError(
        "NumSim's TVM Rust frontend is missing; install tirx-harness with "
        "python -m pip install --no-deps --no-build-isolation /path/to/TIRx-harness"
    )


@cache
def _library() -> Any:
    import tvm_ffi

    library = tvm_ffi.load_module(str(library_path()))
    actual = str(library["numsim_frontend_identity"]())
    expected = _expected_identity()
    if actual != expected:
        raise NumSimBuildError(
            f"NumSim TVM Rust frontend mismatch: expected {expected}, found {actual}"
        )
    try:
        from tvm.backend.cuda.tile_primitive.copy_async.tcgen05_cp import _build_plan

        from .ptx_dialect import register_native_ptx_decoder

        register_native_ptx_decoder(library["numsim_ptx_payload_schema"]())
        tvm_ffi.register_global_func(
            NATIVE_TCGEN05_CP_PLAN, partial(_tcgen05_cp_plan_error, _build_plan), override=True
        )
    except Exception as error:
        raise NumSimBuildError("NumSim native callback registration failed") from error
    return library


def _expected_identity() -> str:
    try:
        return library_path().with_name("_tvm_rust_ext.identity").read_text().strip()
    except OSError as error:
        raise NumSimBuildError(
            "NumSim native build identity is missing; reinstall tirx-harness"
        ) from error


NATIVE_TCGEN05_CP_PLAN = "numsim.frontend.tcgen05_cp_plan_error"


def _tcgen05_cp_plan_error(build_plan: Any, node: Any) -> str:
    """`tile_forms._validate_tcgen_cp_layout`: the production plan's rejection, or ``""``."""

    try:
        build_plan(node)
    except (TypeError, ValueError) as error:
        return str(error)
    return ""


def layout_linear_offsets(layout: Any, count: int) -> Any:
    """Map a bounded logical range in one native call."""
    return _library()["numsim_layout_linear_offsets"](layout, count)


def post_order_nodes(node: Any) -> Any:
    return _library()["numsim_post_order_nodes"](node)


def normalize_host_tensor_maps(func: Any) -> Any:
    import tvm

    return _call("numsim_normalize_host_tensor_maps", func, tvm.__version__)


def semantic_ir_json(value: Any) -> str:
    import tvm

    return str(_library()["numsim_semantic_ir_json"](value, tvm.__version__))


def ptx_call_parts(call: Any, modifier_count: int) -> Any:
    """Read the op, operands, trailing strings, span and result type in one call."""
    return _library()["numsim_ptx_call_parts"](call, modifier_count)


def _failure_data(value: Any) -> Any:
    """Restore ordinary containers from a structured FFI error payload."""

    from tvm_ffi.container import Array, Map

    if isinstance(value, Map):
        return {str(key): _failure_data(item) for key, item in value.items()}
    if isinstance(value, Array):
        return [_failure_data(item) for item in value]
    return value


def raise_failure(payload: dict[str, Any]) -> None:
    """Raise one structured frontend failure as NumSim's own exception class."""

    message = str(payload.get("message", ""))
    if payload.get("kind") == "not_covered":
        raise NumSimBuildError(f"native frontend has no rule for this input: {message}") from None
    if payload.get("kind") == "unmodeled":
        from .source_map import deserialize_source_span

        unmodeled = UnmodeledTIRxFormError(str(payload.get("target", "")), message)
        unmodeled.source_span = deserialize_source_span(payload.get("span"))
        raise unmodeled from None
    if payload.get("kind") == "unsupported":
        raise UnsupportedTIRxError(
            message, unsupported=tuple(payload.get("unsupported", ()))
        ) from None

    raise NumSimBuildError(f"unknown native frontend failure kind: {payload.get('kind')!r}")


def _call(name: str, *args: Any) -> Any:
    """Read a native result envelope without interpreting exception messages."""

    result = _library()[name](*args)
    if result["error"] is not None:
        raise_failure(_failure_data(result["error"]))
    return result["value"]


def _schema_payload() -> dict[str, Any]:
    """The tables the frontend shares with Python, which stays their source."""

    from ..dtype_abi import vector_dtype_abis
    from .ptx_dialect import PTX_SCHEMA_BY_OP_NAME

    return {
        # Every target PTX operation requires an explicit Rust registry row.
        "ptx_table_names": sorted(PTX_SCHEMA_BY_OP_NAME),
        "vector_dtype_abis": {
            abi.dtype: [abi.element_dtype, abi.lanes, abi.element_bits, abi.total_bits]
            for abi in vector_dtype_abis()
        },
    }


@cache
def compile_schema() -> Any:
    """Export the frontend's closed tables once."""

    import tvm_ffi

    return tvm_ffi.convert(_schema_payload())


@cache
def registry() -> dict[str, Any]:
    """The operation registry as Rust owns it (`frontend-rs/src/registry.rs`)."""

    return json.loads(str(_call("numsim_registry", compile_schema())))


def registry_ops() -> tuple[dict[str, Any], ...]:
    """Return the canonical emitting rows in name order."""
    return tuple(registry()["ops"])


def analyze_module(funcs: Any) -> tuple[dict[str, Any], list[list[Any]], dict[str, Any]]:
    """Return a kernel sequence's manifest, post-order nodes and host ABI."""

    payload = _call("numsim_analyze_module", list(funcs), compile_schema())
    return (
        json.loads(str(payload[0])),
        [list(nodes) for nodes in payload[1]],
        json.loads(str(payload[2])),
    )


def host_abi(manifest: dict[str, Any]) -> dict[str, Any]:
    """The host binding facts ``manifest`` determines, or the rejection it carries."""

    return json.loads(str(_call("numsim_host_abi", json.dumps(manifest))))


def default_split_thresholds() -> dict[str, int]:
    defaults = _library()["numsim_split_thresholds"]()
    return {str(name): int(value) for name, value in defaults.items()}


def _split_thresholds() -> dict[str, int]:
    from . import suspend_scaffold

    return suspend_scaffold.thresholds()


def launch_topology(func: Any, environment: dict[Any, Any]) -> dict[str, int]:
    """One kernel's launch topology under concrete scalar parameter values."""

    import tvm_ffi

    payload = _call(
        "numsim_launch_topology", func, compile_schema(), tvm_ffi.convert(dict(environment))
    )
    return json.loads(str(payload))


def compile_module(
    funcs: Any, flags: dict[str, Any]
) -> tuple[dict[str, Any], list[list[Any]], dict[str, Any], str | None, dict[str, Any] | None]:
    """Analyze and emit a kernel sequence in one native call.

    Returns the manifest, each kernel's post-order nodes, the module host ABI,
    the module template (``None`` when a kernel is unsupported or emission
    failed) and the emission failure payload for :func:`raise_failure`.
    """

    import tvm_ffi

    payload = _call(
        "numsim_compile_module",
        list(funcs),
        compile_schema(),
        tvm_ffi.convert({**flags, "split_thresholds": _split_thresholds()}),
    )
    template = None if payload[3] is None else str(payload[3])
    failure = None if payload[4] is None else _failure_data(payload[4])
    return (
        json.loads(str(payload[0])),
        [list(nodes) for nodes in payload[1]],
        json.loads(str(payload[2])),
        template,
        failure,
    )
