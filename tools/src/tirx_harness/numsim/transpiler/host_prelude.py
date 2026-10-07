"""Normalize host-created TensorMaps through the native frontend."""

from __future__ import annotations

from typing import Any

from tvm.tirx import PrimFunc

from .native_frontend import normalize_host_tensor_maps


def normalize_host_tensor_map_prelude(func: PrimFunc) -> PrimFunc:
    """Promote the host TensorMap prelude to implicit typed parameters."""
    if not isinstance(func, PrimFunc):
        raise TypeError(f"expected tvm.tirx.PrimFunc, got {type(func).__name__}")
    return normalize_host_tensor_maps(func)


def normalize_transpile_source(source: Any) -> Any:
    """Normalize one PrimFunc or a sequence without changing its container contract."""
    if isinstance(source, list):
        return [normalize_host_tensor_map_prelude(func) for func in source]
    if isinstance(source, tuple):
        return tuple(normalize_host_tensor_map_prelude(func) for func in source)
    return normalize_host_tensor_map_prelude(source)
