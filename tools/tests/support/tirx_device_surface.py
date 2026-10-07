"""Closed-world walker for the public ``T.cuda`` and ``T.ptx`` surfaces."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from tvm.backend.cuda.ptx.table import TABLE
from tvm.script import tirx as T
from tvm.tirx.op import _canonical_device_intrin_name


TIRX_BUILDER_MODULE = type(T.cuda).__module__
CUDA_PRIVATE_SURFACES = frozenset(
    {"__activemask", "__shfl_sync", "__shfl_up_sync", "__shfl_down_sync", "__shfl_xor_sync"}
)
# ``addr`` is a public nested-address helper, not a target-table instruction
# family.  Keep it in the walked surface while excluding it from the table
# family equality check below.
PTX_NESTED_HELPERS = frozenset({"addr"})
SURFACE_OVERRIDES = {
    "cuda.warp_sum": ("tirx.cuda.warp_reduce",),
    "cuda.warp_max": ("tirx.cuda.warp_reduce",),
    "cuda.warp_min": ("tirx.cuda.warp_reduce",),
    "cuda.cta_sum": ("tirx.cuda.cta_reduce",),
    "cuda.cta_max": ("tirx.cuda.cta_reduce",),
    "cuda.cta_min": ("tirx.cuda.cta_reduce",),
    "cuda.sm100_2sm_leader_smem_addr": (
        "tirx.cuda.cvta_generic_to_shared",
    ),
    "ptx.mbarrier.arrive.cluster_count": ("tirx.ptx.mbarrier_arrive",),
    "ptx.cp_async.legacy": ("tirx.ptx.cp_async",),
    "ptx.cp_async.mbarrier.arrive.noinc": ("tirx.ptx.cp_async_mbarrier_arrive",),
}


def canonical_device_op(raw_name: str) -> str:
    return _canonical_device_intrin_name(f"tirx.{raw_name}")


def surface_ops(path: str, value: Any) -> tuple[str, ...]:
    override = SURFACE_OVERRIDES.get(path)
    if override is not None:
        return override
    leaf = path.rsplit(".", 1)[-1]
    if path.startswith("cuda.") and leaf in CUDA_PRIVATE_SURFACES:
        return (f"tirx.cuda.{leaf}",)
    raw_name = getattr(value, "__tir_op_name__", None)
    if raw_name is None:
        raise AssertionError(f"public TIRx surface {path} has no canonical-op metadata")
    return (canonical_device_op(raw_name),)


def is_tirx_namespace(value: Any) -> bool:
    """Recognize namespace instances owned by the TIRx script builder."""

    return (
        not isinstance(value, type)
        and type(value).__module__ == TIRX_BUILDER_MODULE
        and hasattr(value, "__dict__")
        and (not callable(value) or getattr(value, "__tir_call_op_name__", None) is not None)
    )


def walk_surface(namespace: Any, prefix: str) -> Iterable[tuple[str, tuple[str, ...]]]:
    """Yield every callable public surface and its canonical IR operation(s)."""

    call_name = getattr(namespace, "__tir_call_op_name__", None)
    if call_name is not None:
        yield prefix, (canonical_device_op(call_name),)
    for name in sorted(dir(namespace)):
        if name.startswith("_") and not (prefix == "cuda" and name in CUDA_PRIVATE_SURFACES):
            continue
        try:
            value = getattr(namespace, name)
        except Exception as error:
            raise AssertionError(
                f"failed to inspect public TIRx surface {prefix}.{name}"
            ) from error
        path = f"{prefix}.{name}"
        if is_tirx_namespace(value):
            yield from walk_surface(value, path)
        elif callable(value):
            yield path, surface_ops(path, value)


def walk_device_surfaces() -> tuple[tuple[str, tuple[str, ...]], ...]:
    """Return the complete public CUDA/PTX surface in stable order."""

    ptx_ops_by_family: dict[str, list[str]] = {}
    for entry in TABLE.values():
        ptx_ops_by_family.setdefault(entry.family, []).append(entry.op_name)

    public_ptx_families = {
        name
        for name in dir(T.ptx)
        if not name.startswith("_") and name not in {"SINK", "pred"} | PTX_NESTED_HELPERS
    }
    table_families = set(ptx_ops_by_family)
    if public_ptx_families != table_families:
        raise AssertionError(
            "public T.ptx families disagree with the target table: "
            f"extra={sorted(public_ptx_families - table_families)}, "
            f"missing={sorted(table_families - public_ptx_families)}"
        )

    ptx_surfaces = tuple(
        (f"ptx.{family}", tuple(sorted(ptx_ops_by_family[family])))
        for family in sorted(table_families)
    )
    nested_surfaces = tuple(
        (f"ptx.{name}", (f"tirx.ptx.{name}",)) for name in sorted(PTX_NESTED_HELPERS)
    )
    return tuple((*walk_surface(T.cuda, "cuda"), *ptx_surfaces, *nested_surfaces))


__all__ = [
    "CUDA_PRIVATE_SURFACES",
    "TIRX_BUILDER_MODULE",
    "canonical_device_op",
    "is_tirx_namespace",
    "surface_ops",
    "walk_device_surfaces",
    "walk_surface",
]
