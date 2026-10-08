from __future__ import annotations

import json

import pytest

from tirx_harness.numsim.errors import UnsupportedTIRxError
from tests.numsim.support.kernels import (
    bound_launch_topology,
    divergent_loop_control,
    divergent_while_control,
    fixed_width_integer_expression_mix,
    lane_add,
    no_op_kernel,
    scalar_expression_mix,
    unsupported_bfloat_cast,
    unsupported_exp,
    warpgroup_scope_coordinates,
)
from tirx_harness.numsim.transpiler import frontend
from tirx_harness.numsim.transpiler.frontend import analyze, verify
from tirx_harness.numsim.transpiler.artifact_template import emit_rust_module
from tvm.script import tirx as T


def test_non_tcgen_kernel_does_not_require_a_tcgen_architecture():
    for arch in ("sm_90a", "sm_100a", "sm_107a", "sm_110a"):
        kernel = lane_add.with_attr("tirx.cuda_arch", arch)
        source = emit_rust_module(analyze(kernel), kernel)
        assert "async fn warp_main" in source


def test_control_source_text_preserves_headers_children_and_cached_metadata():
    from tvm_ffi import structural_hash

    before = structural_hash(divergent_loop_control)
    spec = analyze(divergent_loop_control)
    assert structural_hash(divergent_loop_control) == before
    entries = spec.kernels[0].source_map
    loop = next(entry for entry in entries if entry.kind == "For")
    assert loop.text == "for step in range(4):\n    ..."
    store = next(entry for entry in entries if entry.kind == "BufferStore")
    assert store.text == str(store.node).strip()
    assert store.span is not None
    restored = frontend.module_spec_from_manifest(json.loads(json.dumps(spec.to_manifest())))
    assert [entry.to_dict() for entry in restored.kernels[0].source_map] == [
        entry.to_dict() for entry in entries
    ]


def test_source_spans_exclude_process_local_source_name_addresses():
    spans = [
        json.dumps(entry.span.to_dict(), sort_keys=True)
        for entry in analyze(lane_add).kernels[0].source_map
        if entry.span
    ]

    assert spans
    assert all("0x" not in span for span in spans)
    assert all("kernels.py" in span for span in spans)


@T.prim_func(check_well_formed=False)
def conflicting_repeated_warp_extents():
    T.device_entry()
    _warp_a = T.warp_id([2])
    _warp_b = T.warp_id([3])
    _lane = T.lane_id([32])


@T.prim_func(check_well_formed=False)
def invalid_warpgroup_warp_extent():
    T.device_entry()
    _warpgroup = T.warpgroup_id([1])
    _warp = T.warp_id_in_wg([3])
    _lane = T.lane_id([32])


@T.prim_func(check_well_formed=False)
def invalid_warpgroup_thread_extent():
    T.device_entry()
    _warpgroup = T.warpgroup_id([1])
    _thread = T.thread_id_in_wg([96])


@T.prim_func(check_well_formed=False)
def conflicting_direct_and_warpgroup_extents():
    T.device_entry()
    _warpgroup = T.warpgroup_id([2])
    _warp = T.warp_id([4])
    _lane = T.lane_id([32])


@T.prim_func(check_well_formed=False)
def direct_warps_with_warpgroup_local_thread_coordinate():
    T.device_entry()
    _warp = T.warp_id([8])
    _thread = T.thread_id_in_wg([128])


@T.prim_func(check_well_formed=False)
def conflicting_cluster_extents():
    T.device_entry()
    _cluster = T.cluster_id([2])
    _global_cta = T.cta_id([6])
    _cluster_cta = T.cta_id_in_cluster([2])
    _warp = T.warp_id([1])
    _lane = T.lane_id([32])


@T.prim_func
def multidimensional_cluster_with_cta_pair():
    T.device_entry()
    _cta_x, _cta_y = T.cta_id_in_cluster([4, 2])
    _pair = T.cta_id_in_pair()
    _lane = T.lane_id([32])


@T.prim_func
def launch_metadata_attr(output: T.Buffer((1,), "int32")):
    T.device_entry()
    T.attr({"tirx.launch_bounds_min_blocks_per_sm": 1})
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    if lane == 0:
        output[0] = 1


@T.prim_func
def unknown_attr(output: T.Buffer((1,), "int32")):
    T.device_entry()
    T.attr({"numsim.unknown_control": 1})
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    if lane == 0:
        output[0] = 1


@T.prim_func
def supported_loop_metadata(output: T.Buffer((8,), "int32")):
    T.device_entry()
    _warp = T.warp_id([1])
    _lane = T.lane_id([32])
    for index in T.serial(2, unroll=False):
        output[index] = index
    for index in T.unroll(2, 4):
        output[index] = index
    for index in T.serial(4, 6, unroll=True):
        output[index] = index
    for index in T.serial(6, 8, unroll=2):
        output[index] = index


@T.prim_func
def unknown_loop_annotation(output: T.Buffer((4,), "int32")):
    T.device_entry()
    _warp = T.warp_id([1])
    _lane = T.lane_id([32])
    for index in T.serial(4, annotations={"numsim.unknown_loop": 1}):
        output[index] = index


@T.prim_func
def vectorized_loop(output: T.Buffer((4,), "int32")):
    T.device_entry()
    _warp = T.warp_id([1])
    _lane = T.lane_id([32])
    for index in T.vectorized(4):
        output[index] = index


@T.prim_func
def parallel_loop(output: T.Buffer((4,), "int32")):
    T.device_entry()
    _warp = T.warp_id([1])
    _lane = T.lane_id([32])
    for index in T.parallel(4):
        output[index] = index


@T.prim_func
def thread_bound_loop(output: T.Buffer((4,), "int32")):
    T.device_entry()
    for index in T.thread_binding(4, thread="threadIdx.x"):
        output[index] = index


def test_no_op_topology_is_one_future_per_warp():
    spec = analyze(no_op_kernel)

    assert spec.topology.clusters == 2
    assert spec.topology.ctas_per_cluster == 1
    assert spec.topology.warps_per_cta == 3
    assert spec.topology.warp_count == 6
    assert spec.unsupported == ()
    verify(spec)


def test_warpgroup_scope_topology_uses_four_warps_per_group():
    spec = analyze(warpgroup_scope_coordinates)

    assert spec.topology.clusters == 1
    assert spec.topology.ctas_per_cluster == 1
    assert spec.topology.warps_per_cta == 8
    assert spec.topology.warps_per_warpgroup == 4
    assert spec.topology.threads_per_warpgroup == 128
    assert spec.unsupported == ()
    verify(spec)


def test_warpgroup_local_coordinate_does_not_imply_one_group_per_cta():
    spec = analyze(direct_warps_with_warpgroup_local_thread_coordinate)

    assert spec.topology.warps_per_cta == 8
    assert spec.topology.warps_per_warpgroup == 4


def test_frontend_manifest_is_deterministic():
    first = analyze(no_op_kernel)
    second = analyze(no_op_kernel)

    assert first.to_manifest() == second.to_manifest()


def test_manifest_round_trips_and_old_vocabulary_is_rejected():
    """The manifest restores itself, and one with a removed field is a cache miss.

    `module_spec_from_manifest` is only ever called on a manifest read back off
    disk, so it has to do two things: round-trip what the frontend records now,
    and turn every artifact cached under a field the frontend has since dropped
    into a rejection rather than a crash -- `build.py` catches
    `(TypeError, ValueError)` there.
    """

    spec = analyze(lane_add)
    restored = frontend.module_spec_from_manifest(spec.to_manifest())
    assert restored.to_manifest() == spec.to_manifest()

    stale = {**spec.to_manifest(), "conformance_observation_table": []}
    with pytest.raises(ValueError):
        frontend.module_spec_from_manifest(stale)


def test_unknown_reachable_nodes_fail_closed():
    spec = analyze(unsupported_exp)

    assert any("Call" in item for item in spec.unsupported)
    with pytest.raises(UnsupportedTIRxError, match="does not yet support"):
        verify(spec)


def test_lane_add_is_in_the_native_codegen_slice():
    spec = analyze(lane_add)

    assert spec.unsupported == ()
    assert spec.topology.warp_count == 4


def test_divergent_loop_control_is_in_the_native_codegen_slice():
    spec = analyze(divergent_loop_control)

    assert spec.unsupported == ()
    assert spec.topology.warp_count == 1


def test_divergent_while_control_is_in_the_native_codegen_slice():
    spec = analyze(divergent_while_control)

    assert spec.unsupported == ()
    assert spec.topology.warp_count == 1


def test_bound_launch_extent_is_resolved_in_statement_order():
    spec = analyze(bound_launch_topology)

    assert spec.unsupported == ()
    assert spec.topology.clusters == 3
    assert spec.topology.ctas_per_cluster == 1
    assert spec.topology.warps_per_cta == 2


def test_constant_if_then_else_launch_extent_only_evaluates_selected_branch():

    @T.prim_func
    def kernel(choose: T.bool, count: T.int32):
        T.device_entry()
        _warp = T.warp_id([T.if_then_else(choose, count, 1)])
        _lane = T.lane_id([32])

    choose, count = kernel.params
    assert analyze(kernel.specialize({choose: False})).topology.warps_per_cta == 1
    assert analyze(kernel.specialize({choose: True, count: 2})).topology.warps_per_cta == 2
    for bindings in ({choose: True}, {count: 2}):
        with pytest.raises(UnsupportedTIRxError, match="launch extent is not statically known"):
            analyze(kernel.specialize(bindings))


def test_cta_pair_rank_does_not_override_cluster_cta_extent():
    spec = analyze(multidimensional_cluster_with_cta_pair)

    assert spec.unsupported == ()
    assert spec.topology.clusters == 1
    assert spec.topology.ctas_per_cluster == 8
    assert spec.topology.warps_per_cta == 1


def test_scalar_expression_nodes_are_in_the_native_codegen_slice():
    spec = analyze(scalar_expression_mix)

    assert spec.unsupported == ()
    assert spec.topology.warp_count == 1
    census = dict(spec.kernels[0].node_census)
    for kind in (
        "Bind",
        "Cast",
        "FloorDiv",
        "FloorMod",
        "Min",
        "Max",
        "Select",
        "And",
        "Or",
        "Not",
    ):
        assert census[kind] > 0


def test_fixed_width_integer_casts_are_in_the_native_codegen_slice():
    spec = analyze(fixed_width_integer_expression_mix)

    assert spec.unsupported == ()
    verify(spec)


def test_bfloat_cast_is_in_the_native_codegen_slice():
    spec = analyze(unsupported_bfloat_cast)

    assert spec.unsupported == ()
    verify(spec)


@pytest.mark.parametrize(
    ("kernel", "message"),
    [
        (conflicting_repeated_warp_extents, "conflicting launch constraints for warps per CTA"),
        (invalid_warpgroup_warp_extent, "warpgroup warp extent must be 4"),
        (invalid_warpgroup_thread_extent, "warpgroup thread extent must be 128"),
        (conflicting_direct_and_warpgroup_extents, "warpgroup extent implies 8"),
        (conflicting_cluster_extents, "global CTA count 6 disagrees"),
    ],
)
def test_conflicting_or_invalid_topology_constraints_fail_closed(kernel, message):
    with pytest.raises(UnsupportedTIRxError, match=message):
        analyze(kernel)


def test_numerically_irrelevant_launch_metadata_is_explicitly_supported():
    spec = analyze(launch_metadata_attr)

    assert spec.unsupported == ()
    verify(spec)


def test_unknown_attr_semantics_fail_closed():
    spec = analyze(unknown_attr)

    assert any("unknown_control" in item for item in spec.unsupported)
    with pytest.raises(UnsupportedTIRxError, match="unknown_control"):
        verify(spec)


def test_serial_and_unrolled_loop_metadata_are_explicitly_supported():
    spec = analyze(supported_loop_metadata)

    assert spec.unsupported == ()
    verify(spec)


def test_unknown_loop_annotations_fail_closed():
    spec = analyze(unknown_loop_annotation)

    assert any("unknown_loop" in item for item in spec.unsupported)
    with pytest.raises(UnsupportedTIRxError, match="unknown_loop"):
        verify(spec)


@pytest.mark.parametrize(
    ("kernel", "kind"),
    [
        (vectorized_loop, "VECTORIZED"),
        (parallel_loop, "PARALLEL"),
        (thread_bound_loop, "THREAD_BINDING"),
    ],
)
def test_non_serial_loop_semantics_are_not_silently_sequentialized(kernel, kind):
    spec = analyze(kernel)

    assert any(kind in item for item in spec.unsupported)
    with pytest.raises(UnsupportedTIRxError, match=kind):
        verify(spec)
