"""Fail-closed TIRx frontend and launch-topology analysis."""

from __future__ import annotations

import json
import re
import threading
import weakref
from collections import Counter
from dataclasses import asdict, dataclass, replace
from functools import cached_property
from typing import Any

from tvm.tirx import PrimFunc
from tvm_ffi import structural_hash

from ..errors import NumSimBuildError, UnsupportedTIRxError
from .host_prelude import normalize_host_tensor_map_prelude
from . import native_frontend
from .native_frontend import post_order_nodes
from .shared_helpers import plain_text as _plain_text
from .source_map import (
    SourceEntry,
    deserialize_source_span,
    source_span_from_node,
)

_ATTACHED_SOURCE_LOCK = threading.Lock()
_ATTACHED_SOURCE_SPECS: weakref.WeakKeyDictionary[Any, dict[tuple[str, bool], Any]] = (
    weakref.WeakKeyDictionary()
)


_WARP_SIZE = 32


@dataclass(frozen=True)
class LaunchTopology:
    clusters: int
    ctas_per_cluster: int
    warps_per_cta: int
    warps_per_warpgroup: int

    @property
    def warp_count(self) -> int:
        return self.clusters * self.ctas_per_cluster * self.warps_per_cta

    @property
    def threads_per_warpgroup(self) -> int:
        return self.warps_per_warpgroup * _WARP_SIZE


@dataclass(frozen=True)
class BufferSpec:
    parameter: str
    name: str
    shape: tuple[str, ...]
    dtype: str
    scope: str
    elem_offset: str
    layout: str


@dataclass(frozen=True)
class ScalarSpec:
    name: str
    dtype: str
    parameter_index: int


@dataclass(frozen=True)
class PointerSpec:
    name: str
    dtype: str
    storage_scope: str
    parameter_index: int


@dataclass(frozen=True)
class TensorMapSpec:
    name: str
    parameter_index: int
    base_buffer: str | None = None
    base_byte_offset: str | int = 0
    dtype: str | None = None
    tma_dtype: str | None = None
    fp4_shared_layout: str | None = None
    global_shape: tuple[Any, ...] = ()
    global_strides: tuple[Any, ...] = ()
    box_shape: tuple[Any, ...] = ()
    element_strides: tuple[Any, ...] = ()
    interleave: str | None = None
    swizzle: str | None = None
    l2_promotion: str | None = None
    fill_mode: str | None = None

    @property
    def implicit(self) -> bool:
        return self.base_buffer is not None


@dataclass(frozen=True)
class PrimFuncSpec:
    name: str
    structural_hash: str
    script: str
    topology: LaunchTopology
    buffers: tuple[BufferSpec, ...]
    scalars: tuple[ScalarSpec, ...]
    pointers: tuple[PointerSpec, ...]
    tensor_maps: tuple[TensorMapSpec, ...]
    # Read from `source_map` by `_derived_manifest_fields`, which the manifest
    # therefore does not record.
    node_census: tuple[tuple[str, int], ...]
    source_map: tuple[SourceEntry, ...]
    requires_implicit_tmem: bool
    uses_dynamic_tmem_lifecycle: bool
    semantic_requirements: tuple[str, ...]
    unsupported: tuple[str, ...]

    def to_manifest(self, *, include_source_spans: bool = True) -> dict[str, Any]:
        return {
            "name": self.name,
            "structural_hash": self.structural_hash,
            "topology": asdict(self.topology),
            "buffers": [asdict(buffer) for buffer in self.buffers],
            "scalars": [asdict(scalar) for scalar in self.scalars],
            "pointers": [asdict(pointer) for pointer in self.pointers],
            "tensor_maps": [asdict(tensor_map) for tensor_map in self.tensor_maps],
            "source_map": [
                entry.to_dict(include_span=include_source_spans) for entry in self.source_map
            ],
            "requires_implicit_tmem": self.requires_implicit_tmem,
            "uses_dynamic_tmem_lifecycle": self.uses_dynamic_tmem_lifecycle,
            "semantic_requirements": list(self.semantic_requirements),
            "unsupported": list(self.unsupported),
        }


@dataclass(frozen=True)
class ModuleSpec:
    kernels: tuple[PrimFuncSpec, ...]

    @cached_property
    def host_abi(self) -> dict[str, Any]:
        """The public host bindings these kernels expose, derived where they are named.

        ``host_abi.build_host_abi`` reads these facts, or raises the rejection
        they carry in place of them.
        """

        return native_frontend.host_abi(self.to_manifest())

    @property
    def unsupported(self) -> tuple[str, ...]:
        return tuple(item for kernel in self.kernels for item in kernel.unsupported)

    @property
    def topology(self) -> LaunchTopology:
        if len(self.kernels) != 1:
            raise ValueError("A multi-kernel module has no single launch topology")
        return self.kernels[0].topology

    def to_manifest(self, *, include_source_spans: bool = True) -> dict[str, Any]:
        return {
            "kernels": [
                kernel.to_manifest(include_source_spans=include_source_spans)
                for kernel in self.kernels
            ],
        }


def _derived_manifest_fields(source_map: list[SourceEntry]) -> dict[str, Any]:
    """The spec fields the manifest leaves to its other entries."""

    census = Counter(entry.kind for entry in source_map)
    return {"node_census": tuple(sorted(census.items()))}


def _primfunc_spec_from_manifest(kernel: dict[str, Any]) -> PrimFuncSpec:
    """Restore one kernel spec from its manifest entry (no in-process nodes)."""

    source_map = []
    for entry in kernel["source_map"]:
        source_map.append(
            SourceEntry(
                op_id=entry["op_id"],
                kind=entry["kind"],
                text=entry["text"],
                span=deserialize_source_span(entry["span"]),
                op_name=entry.get("op_name"),
            )
        )
    return PrimFuncSpec(
        name=kernel["name"],
        structural_hash=kernel["structural_hash"],
        script="",
        topology=LaunchTopology(**kernel["topology"]),
        buffers=tuple(
            BufferSpec(**{**item, "shape": tuple(item["shape"])})
            for item in kernel["buffers"]
        ),
        scalars=tuple(ScalarSpec(**item) for item in kernel["scalars"]),
        pointers=tuple(PointerSpec(**item) for item in kernel["pointers"]),
        tensor_maps=tuple(
            TensorMapSpec(
                **{
                    **item,
                    **{
                        field_name: tuple(item.get(field_name, ()))
                        for field_name in (
                            "global_shape",
                            "global_strides",
                            "box_shape",
                            "element_strides",
                        )
                    },
                }
            )
            for item in kernel["tensor_maps"]
        ),
        source_map=tuple(source_map),
        requires_implicit_tmem=kernel["requires_implicit_tmem"],
        uses_dynamic_tmem_lifecycle=kernel["uses_dynamic_tmem_lifecycle"],
        semantic_requirements=tuple(kernel["semantic_requirements"]),
        unsupported=tuple(kernel["unsupported"]),
        **_derived_manifest_fields(source_map),
    )


def module_spec_from_manifest(value: Any) -> ModuleSpec:
    """Restore the runtime-facing spec cached beside one generated artifact."""

    if not isinstance(value, dict) or not isinstance(value.get("kernels"), list):
        raise ValueError("module spec manifest must contain a kernel list")
    kernels: list[PrimFuncSpec] = []
    try:
        for kernel in value["kernels"]:
            kernels.append(_primfunc_spec_from_manifest(kernel))
    except (AttributeError, KeyError, TypeError, ValueError) as error:
        raise ValueError(f"invalid module spec manifest: {error}") from error
    result = ModuleSpec(kernels=tuple(kernels))
    canonical = json.dumps(result.to_manifest(), sort_keys=True, separators=(",", ":"))
    supplied = json.dumps(value, sort_keys=True, separators=(",", ":"))
    if canonical != supplied:
        raise ValueError("module spec manifest contains non-canonical field values")
    return result


def _substituted_source(spec: ModuleSpec, kernels: tuple[PrimFuncSpec, ...]) -> ModuleSpec:
    """``spec`` with ``kernels`` that differ from its own in source information only.

    Host ABI depends on parameter bindings, which this substitution preserves.
    Reuse that memo without deriving it again. A plain ``replace()`` can change
    a semantic field and therefore does not inherit it.
    """

    substituted = replace(spec, kernels=kernels)
    if "host_abi" in spec.__dict__:
        object.__setattr__(substituted, "host_abi", spec.__dict__["host_abi"])
    return substituted


def attach_source_nodes(
    spec: ModuleSpec,
    source: Any,
    *,
    _render_script: bool = True,
    _cache_key: str | None = None,
) -> ModuleSpec:
    """Attach source nodes, reusing the caller's compile or artifact cache key.

    The optional key already identifies semantic IR and code generation. Keep
    the input spec beside the attachment so changed metadata cannot reuse a
    previous attachment, even when a caller supplies the same key.
    """

    sources = tuple(source) if isinstance(source, (list, tuple)) else (source,)
    funcs = tuple(normalize_host_tensor_map_prelude(func) for func in sources)
    if len(funcs) != len(spec.kernels):
        raise ValueError("cached module spec does not match the source kernel count")
    cache_key = None if _cache_key is None else (_cache_key, _render_script)
    if cache_key is not None and len(funcs) == 1:
        # Normalization returns a temporary Python wrapper. Key the weak cache
        # by the caller-owned source so the entry can outlive this call.
        with _ATTACHED_SOURCE_LOCK:
            cached = _ATTACHED_SOURCE_SPECS.get(sources[0], {}).get(cache_key)
        if cached is not None and cached[0] == spec:
            return cached[1]
    kernels: list[PrimFuncSpec] = []
    for kernel, func in zip(spec.kernels, funcs, strict=True):
        if str(structural_hash(func)) != kernel.structural_hash:
            raise ValueError("cached module spec does not match the source structural hash")
        nodes = post_order_nodes(func.body)
        if len(nodes) != len(kernel.source_map):
            raise ValueError("cached module spec does not match the source operation count")
        source_map = []
        for index, (entry, node) in enumerate(zip(kernel.source_map, nodes, strict=True)):
            if type(node).__name__ != entry.kind:
                raise ValueError(
                    "cached module spec source kind does not match the source operation "
                    f"at entry {index}: expected {entry.kind}, found {type(node).__name__}"
                )
            source_map.append(replace(entry, span=source_span_from_node(node), node=node))
        rebound_source_map = tuple(source_map)
        kernels.append(
            replace(
                kernel,
                script=_plain_text(func.script()) if _render_script else "",
                source_map=rebound_source_map,
            )
        )
    attached = _substituted_source(spec, tuple(kernels))
    if cache_key is not None and len(funcs) == 1:
        with _ATTACHED_SOURCE_LOCK:
            by_identity = _ATTACHED_SOURCE_SPECS.setdefault(sources[0], {})
            by_identity[cache_key] = (spec, attached)
    return attached


def source_kernels(func: Any) -> tuple[PrimFunc, ...]:
    """The normalized PrimFunc sequence of one module."""

    funcs = tuple(func) if isinstance(func, (list, tuple)) else (func,)
    if not funcs:
        raise ValueError("NumSim cannot transpile an empty kernel sequence")
    kernels = []
    for item in funcs:
        if not isinstance(item, PrimFunc):
            raise TypeError(f"NumSim expects tvm.tirx.PrimFunc, got {type(item).__name__}")
        kernels.append(normalize_host_tensor_map_prelude(item))
    return tuple(kernels)


def native_module_spec(
    funcs: tuple[PrimFunc, ...],
    manifest: dict[str, Any],
    nodes: list[list[Any]],
    host_abi: dict[str, Any],
    *,
    render_script: bool,
) -> ModuleSpec:
    """Bind a native module manifest and its host ABI to the source kernels."""

    bound = []
    for func, kernel, kernel_nodes in zip(funcs, manifest["kernels"], nodes, strict=True):
        spec = _primfunc_spec_from_manifest(kernel)
        if len(kernel_nodes) != len(spec.source_map):
            raise NumSimBuildError("native frontend returned a source map without its nodes")
        source_map = tuple(
            replace(entry, node=node)
            for entry, node in zip(spec.source_map, kernel_nodes, strict=True)
        )
        bound.append(
            replace(
                spec,
                source_map=source_map,
                script=_plain_text(func.script()) if render_script else "",
            )
        )
    result = ModuleSpec(kernels=tuple(bound))
    # The native frontend named this very manifest, so the memo needs no second
    # native call to confirm it.
    object.__setattr__(result, "host_abi", host_abi)
    return result


def native_manifest_mismatch(manifest: dict[str, Any], spec: ModuleSpec) -> int | None:
    """The index of the first kernel ``spec`` describes differently from ``manifest``."""

    for index, (kernel, kernel_spec) in enumerate(
        zip(manifest["kernels"], spec.kernels, strict=True)
    ):
        if _primfunc_spec_from_manifest(kernel).to_manifest() != kernel_spec.to_manifest():
            return index
    return None


def _extract_topology(
    func: PrimFunc,
    environment: dict[Any, int | float | bool] | None = None,
) -> LaunchTopology:
    """The launch topology of ``func`` under concrete scalar parameter values."""

    return LaunchTopology(**native_frontend.launch_topology(func, environment or {}))


def analyze(
    func: PrimFunc | list[PrimFunc] | tuple[PrimFunc, ...],
    *,
    _render_script: bool = True,
) -> ModuleSpec:
    funcs = source_kernels(func)
    manifest, nodes, host_abi = native_frontend.analyze_module(funcs)
    return native_module_spec(funcs, manifest, nodes, host_abi, render_script=_render_script)


def _first_unsupported_span(spec: ModuleSpec) -> Any | None:
    """Locate the node behind the first unsupported entry.

    Unsupported entries are recorded as ``op#<id>:<reason>`` while the analysis
    still knows which operation produced them, and the kernel's source map keeps
    each operation's parser span. Resolving the pair here means a frontend
    finding points at the offending node instead of at the enclosing function.
    """

    for kernel in spec.kernels:
        for item in kernel.unsupported:
            span = _locate_unsupported_entry(kernel, item)
            if span is not None:
                return span
    return None


def _locate_unsupported_entry(kernel: Any, item: str) -> Any | None:
    """Resolve one ``op#<id>:`` or ``buffer|pointer:<name>:`` entry to a span."""

    op_match = re.match(r"op#(\d+):", item)
    if op_match is not None:
        op_id = int(op_match.group(1))
        for entry in kernel.source_map:
            if entry.op_id == op_id and entry.span is not None:
                return entry.span
        return None

    named = re.match(r"(?:buffer|pointer):([^:]+):", item)
    if named is None:
        return None
    # A buffer-level rejection names no operation, so point at the first access
    # that touches the offending buffer. ``node`` is the live IR object, which
    # keeps this an identity match rather than a text search; it is absent for
    # a spec restored from cache, in which case this entry stays unlocated.
    name = named.group(1)
    for entry in kernel.source_map:
        node = entry.node
        if entry.span is None or node is None:
            continue
        buffer = getattr(node, "buffer", None)
        if buffer is not None and _plain_text(buffer.name) == name:
            return entry.span
    return None


def verify(spec: ModuleSpec) -> None:
    if not spec.unsupported:
        return
    details = ", ".join(spec.unsupported[:20])
    if len(spec.unsupported) > 20:
        details += f", ... ({len(spec.unsupported)} total)"
    raise UnsupportedTIRxError(
        f"NumSim does not yet support all reachable TIRx nodes: {details}",
        unsupported=spec.unsupported,
        source_span=_first_unsupported_span(spec),
    )
