"""Native module emission (`emit/module.rs`) behind the historical entry point."""

from __future__ import annotations

from typing import Any, Literal

from tvm_ffi import structural_hash as tvm_structural_hash

from ..abi import NUMSIM_ABI_VERSION
from ..errors import NumSimBuildError, UnsupportedTIRxError
from . import native_frontend
from .frontend import ModuleSpec, native_manifest_mismatch, verify
from .host_prelude import normalize_host_tensor_map_prelude


def _normalize_funcs(func: Any) -> tuple[Any, ...]:
    return tuple(func) if isinstance(func, (list, tuple)) else (func,)


def compile_native_module(
    funcs: tuple[Any, ...],
    *,
    analysis_capable: bool,
    analysis_checker: str | None,
) -> tuple[
    dict[str, Any], list[list[Any]], str, dict[str, Any], str | None, dict[str, Any] | None
]:
    """Analyze and emit normalized kernels in one native call."""

    return native_frontend.compile_module(
        funcs,
        {
            "analysis_capable": analysis_capable,
            "analysis_checker": "" if analysis_checker is None else analysis_checker,
            "numsim_abi_version": NUMSIM_ABI_VERSION,
        },
    )


def render_native_module(compiled: Any, spec: ModuleSpec) -> str:
    """The generated module of ``spec``: host ABI errors precede emission errors."""

    _manifest, _nodes, host_abi, template, failure = compiled
    if "error" in host_abi:
        raise UnsupportedTIRxError(f"NumSim module has an invalid host ABI: {host_abi['error']}")
    if failure is not None:
        native_frontend.raise_failure(failure)
    if template is None:
        verify(spec)
        raise NumSimBuildError("native frontend emitted no module for a supported specification")
    return template


def emit_rust_module(
    spec: ModuleSpec,
    func: Any,
    *,
    analysis_capable: bool = False,
    analysis_checker: Literal["synccheck", "racecheck"] | None = None,
    validate_spec: bool = True,
    verify_spec: bool = True,
) -> str:
    """Emit a native Rust artifact; no TIRx node survives this call."""

    if analysis_checker not in {None, "synccheck", "racecheck"}:
        raise ValueError(f"unknown native analysis checker: {analysis_checker!r}")
    if analysis_checker is not None and not analysis_capable:
        raise ValueError("a native analysis checker requires analysis-capable code generation")

    funcs = tuple(normalize_host_tensor_map_prelude(item) for item in _normalize_funcs(func))
    if len(funcs) != len(spec.kernels):
        raise UnsupportedTIRxError(
            "NumSim module specification and PrimFunc sequence have different lengths"
        )
    if validate_spec:
        for index, (kernel_func, kernel_spec) in enumerate(zip(funcs, spec.kernels, strict=True)):
            structural_hash = str(tvm_structural_hash(kernel_func))
            if structural_hash != kernel_spec.structural_hash:
                raise UnsupportedTIRxError(
                    "NumSim module specification does not describe PrimFunc "
                    f"{index}: expected structural hash {kernel_spec.structural_hash}, "
                    f"got {structural_hash}"
                )
    compiled = compile_native_module(
        funcs,
        analysis_capable=analysis_capable,
        analysis_checker=analysis_checker,
    )
    mismatch = native_manifest_mismatch(compiled[0], spec)
    if mismatch is not None:
        if validate_spec:
            raise UnsupportedTIRxError(
                "NumSim module specification does not describe PrimFunc "
                f"{mismatch}: semantic/ABI manifest differs"
            )
        raise NumSimBuildError(
            "native frontend has no rule for this input: "
            "kernel manifest does not match the native analysis"
        )
    if verify_spec:
        verify(spec)
    return render_native_module(compiled, spec)


__all__ = ["emit_rust_module"]
