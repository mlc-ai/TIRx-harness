from __future__ import annotations

import pytest
from tvm import ir

from tests.numsim.support.kernels import raw_scalar_call_mix, tile_copy_cast_mul
from tests.numsim.support.manifest import call_op_names
from tests.support.tirx_device_surface import (
    TIRX_BUILDER_MODULE,
    is_tirx_namespace,
    walk_device_surfaces,
    walk_surface,
)
from tirx_harness.numsim.transpiler import native_frontend
from tirx_harness.numsim.transpiler.frontend import analyze
from tirx_harness.numsim.transpiler.ptx_dialect import PTX_SCHEMA_BY_OP_NAME

_DEVICE_PREFIXES = ("tirx.cuda.", "tirx.ptx.")

# These register-only forms require the unsupported WGMMA execution model.
_INTENTIONAL_REGISTER_ONLY_PTX_REJECTIONS = {
    "tirx.ptx.wgmma_b1_rs",
    "tirx.ptx.wgmma_b1_ss",
    "tirx.ptx.wgmma_int_rs",
    "tirx.ptx.wgmma_int_ss",
}


def test_surface_walk_fails_closed_on_public_attribute_errors():
    class BrokenSurface:
        def __dir__(self):
            return ["broken"]

        @property
        def broken(self):
            raise RuntimeError("broken wrapper")

    with pytest.raises(AssertionError, match="failed to inspect public TIRx surface root.broken"):
        list(walk_surface(BrokenSurface(), "root"))


def test_namespace_detection_uses_builder_ownership_not_class_name_suffix():
    class ExplicitSurface:
        child = staticmethod(lambda: None)

    ExplicitSurface.__module__ = TIRX_BUILDER_MODULE
    namespace = ExplicitSurface()
    namespace.child.__tir_op_name__ = "cuda_elect_sync"

    assert is_tirx_namespace(namespace)
    assert list(walk_surface(namespace, "synthetic")) == [
        ("synthetic.child", ("tirx.cuda.elect_sync",))
    ]


def _authoritative_device_ops() -> set[str]:
    return {name for name in ir.Op.list_op_names() if name.startswith(_DEVICE_PREFIXES)}


def test_registry_covers_tirx_cuda_and_ptx_ops():
    authoritative = _authoritative_device_ops()
    registered = {
        spec["ir_name"]
        for spec in native_frontend.registry_ops()
        if spec["ir_name"].startswith(_DEVICE_PREFIXES)
    }

    assert authoritative <= registered


def test_registry_uses_current_log2_namespace():
    registered = {spec["ir_name"] for spec in native_frontend.registry_ops()}
    assert ir.Op.get("prim.log2").name in registered
    assert "tirx.log2" not in registered


def test_registry_retains_reviewed_ops_absent_from_target_table():
    import json

    import tvm_ffi

    missing = "tirx.ptx.add_mixed_vec_up"
    schema = native_frontend._schema_payload()
    assert missing in schema["ptx_table_names"]
    schema["ptx_table_names"] = [name for name in schema["ptx_table_names"] if name != missing]

    rebuilt = {
        row["ir_name"]
        for row in json.loads(
            str(native_frontend._call("numsim_registry", tvm_ffi.convert(schema)))
        )["ops"]
    }

    assert missing in rebuilt
    assert set(schema["ptx_table_names"]) <= rebuilt


def test_register_only_ptx_rejections_are_intentionally_nonlocal():
    specs = {spec["ir_name"]: spec for spec in native_frontend.registry_ops()}
    register_only = {
        name
        for name, entry in PTX_SCHEMA_BY_OP_NAME.items()
        if name.startswith("tirx.ptx.")
        and not entry.orders_memory
        and all(operand.kind == "reg" for operand in entry.operands)
    }

    rejected = {name for name in register_only if specs[name]["support"] == "rejected"}
    assert rejected == _INTENTIONAL_REGISTER_ONLY_PTX_REJECTIONS


def test_every_public_cuda_and_ptx_surface_maps_to_registered_semantics():
    surfaces = walk_device_surfaces()
    registered = {spec["ir_name"] for spec in native_frontend.registry_ops()}
    missing = {
        path: tuple(op for op in operations if op not in registered)
        for path, operations in surfaces
        if any(op not in registered for op in operations)
    }
    assert missing == {}


def test_explicit_rejections_have_exact_public_reasons():
    rejected = {
        spec["ir_name"]: spec["reason"]
        for spec in native_frontend.registry_ops()
        if spec["support"] == "rejected"
    }
    helper = next(
        row for row in native_frontend.registry_ops() if row["ir_name"] == "tirx.cuda.func_call"
    )
    assert helper["family"] == "validated_cuda_helper"
    assert helper["support"] == "modeled"
    assert "opaque CUDA helper bodies are rejected" in helper["reason"]
    wgmma_reason = "WGMMA is not supported by the SM100 NumSim target"
    authoritative_wgmma = {
        name
        for name in _authoritative_device_ops()
        if name.startswith(("tirx.cuda.wgmma_", "tirx.ptx.wgmma_"))
    }
    assert {name for name, reason in rejected.items() if reason == wgmma_reason} == (
        authoritative_wgmma
    )

    unreviewed_reason = "target-table PTX operation has no reviewed NumSim semantics"
    unreviewed = {name for name, reason in rejected.items() if reason == unreviewed_reason}
    assert unreviewed
    assert unreviewed <= set(PTX_SCHEMA_BY_OP_NAME)
    reviewed_rejections = set(rejected) - authoritative_wgmma - unreviewed
    assert reviewed_rejections == set()


def test_timing_and_observation_surfaces_declare_representative_fidelity():
    support = {spec["ir_name"]: spec["support"] for spec in native_frontend.registry_ops()}
    for name in (
        "tirx.cuda.clock64",
        "tirx.cuda.nano_sleep",
        "tirx.cuda.printf",
        "tirx.cuda.elect_sync",
        "tirx.ptx.ex2",
        "tirx.ptx.mbarrier_try_wait_parity",
        "tirx.ptx.mbarrier_try_wait_parity_no_hint",
        "tirx.ptx.rcp",
    ):
        assert support[name] == "deterministic_representative"
    for name in (
        "tirx.cuda.thread_fence",
        "tirx.ptx.fence",
        "tirx.ptx.griddepcontrol",
    ):
        assert support[name] == "ordering_only"


def test_registry_entries_have_complete_audit_metadata():
    for spec in native_frontend.registry_ops():
        assert set(spec) == {"ir_name", "family", "support", "reason", "suspends"}
        assert isinstance(spec["suspends"], bool)
        assert spec["family"]
        if spec["support"] == "rejected":
            assert spec["reason"]


def test_call_registry_owns_each_canonical_operation_once():
    specs = [spec for spec in native_frontend.registry_ops() if spec["support"] != "rejected"]

    assert specs
    assert len({spec["ir_name"] for spec in specs}) == len(specs)
    by_name = {spec["ir_name"]: spec for spec in specs}
    for path in native_frontend.registry()["contextual_op_paths"]:
        assert path["op"]["ir_name"] in by_name


def test_cuda_atomic_form_metadata_distinguishes_the_ordinary_packed_abi():
    specs = {spec["ir_name"]: spec for spec in native_frontend.registry_ops()}
    assert "one atomic RMW per component" in (specs["tirx.cuda.atomic_add"]["reason"] or "")
    cas_reason = specs["tirx.cuda.atomic_cas"]["reason"] or ""
    assert "whole-value bitwise" in cas_reason
    assert "bool vectors" in cas_reason
    assert "float4/float6" in cas_reason


def test_frontend_manifest_records_the_ops_each_kernel_calls():
    scalar = analyze(raw_scalar_call_mix).kernels[0]
    tile = analyze(tile_copy_cast_mul).kernels[0]

    assert {
        "tirx.cuda.thread_rank",
        "tirx.cuda.make_float2",
        "tirx.cuda.elect_sync",
        "tirx.ptx.ex2",
    } <= call_op_names(scalar)
    assert {
        str(entry.node.op.name) for entry in tile.source_map if entry.kind == "TilePrimitiveCall"
    } == {"tirx.tile.cast", "tirx.tile.copy", "tirx.tile.mul"}


def test_unknown_target_op_requires_an_explicit_registry_row():
    import tvm_ffi

    schema = native_frontend._schema_payload()
    schema["ptx_table_names"].append("tirx.ptx.unreviewed_test_instruction")
    with pytest.raises(ValueError, match="has no reviewed registry row"):
        native_frontend._call("numsim_registry", tvm_ffi.convert(schema))
