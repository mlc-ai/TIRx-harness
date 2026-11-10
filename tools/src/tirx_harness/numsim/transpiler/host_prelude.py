"""Normalize kernel sources: expand tile operations and host-created TensorMaps."""

from __future__ import annotations

import json
from typing import Any

import tvm_ffi

from tvm_tirx import ir
from tvm_tirx.tirx import Function

from .native_frontend import normalize_host_tensor_maps

_TILE_OPERATIONS = frozenset({"tirx.cuda.tile.apply", "tirx.cuda.tile.ewise"})
# NumSim simulates SM100; a kernel's backend_config may select a later architecture.
_DEFAULT_CUDA_ARCH = "sm_100a"


def _cuda_arch(func: Function) -> str:
    """``backend_config["cuda"]["arch"]`` of the device entry, else of ``func``."""
    configs = [
        statement.attrs["backend_config"]
        for statement in (func.body.seq if func.body is not None else ())
        if isinstance(statement, ir.RegionStmt)
        and str(statement.op.name) in {"tirx.device_entry", "tirx.device_scope"}
        and "backend_config" in statement.attrs
    ]
    if func.attrs is not None and "backend_config" in func.attrs:
        configs.append(func.attrs["backend_config"])
    for config in configs:
        arch = json.loads(str(config)).get("cuda", {}).get("arch")
        if arch is not None:
            return str(arch)
    return _DEFAULT_CUDA_ARCH


def dispatch_tile_operations(func: Function) -> Function:
    """Expand CUDA tile operations into the instructions TIRx's TileDispatch selects."""
    found = []

    def visit(call: ir.Call) -> None:
        if isinstance(call.op, ir.Op) and str(call.op.name) in _TILE_OPERATIONS:
            found.append(call)

    if func.body is not None:
        tvm_ffi.structural_walk(func.body, (ir.Call, visit))
    if not found:
        return func
    import tvm_tirx

    target = tvm_tirx.target.Target({"kind": "cuda", "arch": _cuda_arch(func)}, host="llvm")
    module = tvm_tirx.IRModule({"main": func.with_attr("target", target)})
    return tvm_tirx.tirx.transform.TileDispatch()(module)["main"]


def normalize_host_tensor_map_prelude(func: Function) -> Function:
    """Expand tile operations, then promote the host TensorMap prelude to implicit
    typed parameters."""
    if not isinstance(func, Function):
        raise TypeError(f"expected tvm_tirx.tirx.Function, got {type(func).__name__}")
    return normalize_host_tensor_maps(dispatch_tile_operations(func))


def normalize_transpile_source(source: Any) -> Any:
    """Normalize one PrimFunc or a sequence without changing its container contract."""
    if isinstance(source, list):
        return [normalize_host_tensor_map_prelude(func) for func in source]
    if isinstance(source, tuple):
        return tuple(normalize_host_tensor_map_prelude(func) for func in source)
    return normalize_host_tensor_map_prelude(source)
