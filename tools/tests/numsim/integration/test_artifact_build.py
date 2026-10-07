from __future__ import annotations

import json

import numpy as np
import pytest
from tvm.ir import load_json, save_json

from tirx_harness import numsim, racecheck
from tirx_harness.numsim.api import ExecutionSubset
from tirx_harness.numsim.abi import abi_metadata
from tirx_harness.numsim.bindings import prepare_bindings
from tests.numsim.support.kernels import (
    bar_sync_after_full_warp_continue,
    bound_dynamic_cta_extent,
    bulk_shared_to_cluster_u64_addresses,
    compose_swizzle_alias,
    cta_sync_after_full_warp_continue,
    divergent_loop_control,
    divergent_while_control,
    dps_float_arithmetic,
    dynamic_for_after_full_warp_continue,
    dynamic_rows,
    elect_sync_integer_branch,
    extent_free_warp_id,
    guarded_if_then_else_load,
    guarded_select_load,
    lane_add,
    local_array_per_lane,
    mapped_remote_mbarrier_pointer,
    matrix_add_2d,
    mbarrier_missing_arrivals,
    mbarrier_phase_reuse,
    mbarrier_remote_cta,
    mbarrier_varying_uniform_phase,
    mbarrier_wait_after_full_warp_continue,
    native_varying_assert,
    nested_mask_parent_scope,
    no_op_kernel,
    overlapping_alias_write,
    parallel_cluster_remote_shared_exchange,
    physical_address_value,
    raw_scalar_call_mix,
    remote_shared_read_and_warp_reduce,
    remote_shared_write_ownership,
    scalar_buffer_types,
    scalar_expression_mix,
    scoped_syncs,
    shared_alias_per_cta,
    shared_uninitialized_read,
    warp_pure_calls,
    warpgroup_scope_coordinates,
)
from tests.numsim.support.remote_mbarrier import (
    mapped_remote_mbarrier_cluster_view,
    mapped_remote_mbarrier_pointer_expect_tx,
)
from tirx_harness.numsim.transpiler.frontend import analyze
from tirx_harness.numsim.transpiler.source_map import flatten_source_span
from tirx_harness.numsim.transpiler.artifact_template import emit_rust_module
from tvm import tirx
from tvm_ffi import structural_equal, structural_hash
from tvm.script import tirx as T
from tvm.tirx.layout import ComposeLayout, S, TileLayout

_PADDED_COMPOSE_LAYOUT = ComposeLayout(0, 0, 0, TileLayout(S[(2, 2) : (4, 1)]))


@T.prim_func
def lazy_logical_buffer_load(source: T.Buffer((1,), "int32"), output: T.Buffer((32,), "int32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    and_value: T.let = (lane == 0) and (source[lane] == 5)
    or_value: T.let = (lane != 0) or (source[lane] == 5)
    output[lane] = T.cast(and_value, "int32") + T.cast(or_value, "int32") * 2


@T.prim_func
def lane_and_thread_scope_ids(output: T.Buffer((4, 32, 2), "int32")):
    T.device_entry()
    warp = T.warp_id([4])
    lane = T.lane_id([32])
    thread = T.thread_id([128])
    output[warp, lane, 0] = lane
    output[warp, lane, 1] = thread


@T.prim_func
def dps_pointer_offset(output: T.Buffer((33,), "float32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    destination = T.ptr_byte_offset(output.ptr_to([lane]), T.uint32(4), "float32")
    T.ptx.st.global_.f32(destination, T.cast(lane, "float32") + T.float32(100))


@T.prim_func
def dps_read_only_destination(output: T.Buffer((33,), "float32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    destination = output.access_ptr("r", offset=lane, extent=1)
    T.ptx.st.global_.f32(destination, T.cast(lane, "float32") + T.float32(100))


@T.prim_func
def dps_destination_after_access_extent(output: T.Buffer((33,), "float32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    bounded = output.access_ptr("w", offset=lane, extent=1)
    destination = T.ptr_byte_offset(bounded, T.uint32(4), "float32")
    T.ptx.st.global_.f32(destination, T.cast(lane, "float32") + T.float32(100))


@T.prim_func
def remote_mbarrier_explicit_count(output: T.Buffer((2,), "int32")):
    T.device_entry()
    _cluster = T.cluster_id([1])
    cta = T.cta_id_in_cluster([2])
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    barriers = T.alloc_buffer((1,), "uint64", scope="shared")
    if (cta == 0) and (lane == 0):
        T.ptx.mbarrier.init.shared.b64(T.address_of(barriers[0]), 4)
    T.ptx.fence.proxy.async_.shared__cluster()
    T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.cluster_sync()
    if lane == 0:
        if cta == 0:
            T.cuda.mbarrier_wait(T.address_of(barriers[0]), 0)
        else:
            remote_barrier = T.alloc_local((1,), "uint64")
            T.ptx.mapa.shared__cluster.u64(remote_barrier[0], barriers.ptr_to([0]), T.uint32(0))
            T.ptx.mbarrier.arrive.b64(remote_barrier[0], T.uint32(4), pred=T.bool(True))
    T.cuda.cluster_sync()
    if lane == 0:
        output[cta] = 1


@T.prim_func
def padded_compose_alias(output: T.Buffer((6,), "uint32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((2, 2), "uint32", scope="shared", layout=_PADDED_COMPOSE_LAYOUT)
    dense = T.decl_buffer((6,), "uint32", data=shared.data, scope="shared")
    if lane == 0:
        dense[2] = 90
        dense[3] = 91
        shared[0, 0] = 10
        shared[0, 1] = 11
        shared[1, 0] = 12
        shared[1, 1] = 13
        for index in T.serial(6):
            output[index] = dense[index]


@T.prim_func
def cta_reductions_preserve_scratch(output: T.Buffer((3, 3), "float32")):
    T.device_entry()
    warp = T.warp_id([2])
    lane = T.lane_id([32])
    sum_scratch = T.alloc_buffer((2,), "float32", scope="shared")
    max_scratch = T.alloc_buffer((2,), "float32", scope="shared")
    min_scratch = T.alloc_buffer((2,), "float32", scope="shared")
    value = T.cast(warp * 100 + lane, "float32")
    sum_result = T.cuda.cta_sum(value, 2, sum_scratch.ptr_to([0]))
    max_result = T.cuda.cta_max(value, 2, max_scratch.ptr_to([0]))
    min_result = T.cuda.cta_min(value, 2, min_scratch.ptr_to([0]))
    if (warp == 0) and (lane == 0):
        output[0, 0] = sum_result
        output[0, 1] = sum_scratch[0]
        output[0, 2] = sum_scratch[1]
        output[1, 0] = max_result
        output[1, 1] = max_scratch[0]
        output[1, 2] = max_scratch[1]
        output[2, 0] = min_result
        output[2, 1] = min_scratch[0]
        output[2, 2] = min_scratch[1]


@T.prim_func
def cuda_float_reduction_edge_bits(
    operands: T.Buffer((2,), "uint32"), output: T.Buffer((7,), "uint32")
):
    T.device_entry()
    warp = T.warp_id([2])
    lane = T.lane_id([32])
    sum_scratch = T.alloc_buffer((2,), "float32", scope="shared")
    max_scratch = T.alloc_buffer((2,), "float32", scope="shared")
    min_scratch = T.alloc_buffer((2,), "float32", scope="shared")
    is_special = (warp == 0) and (lane == 1)
    nan_value = T.if_then_else(
        is_special, T.reinterpret("float32", T.uint32(0x7FC12345)), T.float32(0.0)
    )
    zero_value = T.if_then_else(is_special, T.float32(0.0), T.float32(-0.0))
    sum_result = T.cuda.cta_sum(nan_value, 2, sum_scratch.ptr_to([0]))
    max_result = T.cuda.cta_max(zero_value, 2, max_scratch.ptr_to([0]))
    min_result = T.cuda.cta_min(zero_value, 2, min_scratch.ptr_to([0]))
    if (warp == 0) and (lane == 0):
        lhs = T.reinterpret("float32", operands[0])
        rhs = T.reinterpret("float32", operands[1])
        output[0] = T.reinterpret("uint32", sum_result)
        output[1] = T.reinterpret("uint32", max_result)
        output[2] = T.reinterpret("uint32", min_result)
        output[3] = T.reinterpret("uint32", T.max(lhs, rhs))
        output[4] = T.reinterpret("uint32", T.max(rhs, lhs))
        output[5] = T.reinterpret("uint32", T.min(lhs, rhs))
        output[6] = T.reinterpret("uint32", T.min(rhs, lhs))


def _encode_bf16(values: np.ndarray) -> np.ndarray:
    bits = np.asarray(values, dtype=np.float32).view(np.uint32)
    rounded = bits + np.uint32(0x7FFF) + ((bits >> np.uint32(16)) & np.uint32(1))
    return (rounded >> np.uint32(16)).astype(np.uint16)


def _decode_bf16(bits: np.ndarray) -> np.ndarray:
    return (np.asarray(bits, dtype=np.uint16).astype(np.uint32) << np.uint32(16)).view(np.float32)


def test_codegen_rejects_a_spec_paired_with_different_primfuncs() -> None:
    spec = analyze((lane_add, matrix_add_2d))

    with pytest.raises(numsim.UnsupportedTIRxError, match="does not describe PrimFunc 0"):
        emit_rust_module(spec, (matrix_add_2d, lane_add))


def test_codegen_rejects_alpha_renamed_host_abi_with_the_same_structural_hash() -> None:
    def make_func(parameter_name: str):
        buffer = tirx.decl_buffer((1,), "int32", name=parameter_name)
        return tirx.PrimFunc([buffer], tirx.Evaluate(tirx.IntImm("int32", 0)))

    original = make_func("original")
    renamed = make_func("renamed")
    assert structural_hash(original) == structural_hash(renamed)
    assert structural_equal(original, renamed)

    with pytest.raises(numsim.UnsupportedTIRxError, match="semantic/ABI manifest differs"):
        emit_rust_module(analyze(original), renamed)


def test_transpiled_artifact_runs_one_future_per_warp(tmp_path, expect_harness_surface):
    module = numsim.transpile(no_op_kernel, cache_dir=tmp_path)
    result = numsim.Engine(max_workers=1).run(module, {})

    assert module.load().metadata()["warp_count"] == 6
    assert result.outputs == {}
    assert result.stats["task_count"] == 6
    assert result.stats["completed_task_count"] == 6
    assert result.stats["poll_order"] == list(range(6))
    assert result.stats["worker_count"] == 1
    assert result.stats["scheduling_domain_count"] == 2
    assert "numsim_emit_helper_items" not in module.rust_source
    assert "numsim_skip_helper_items" not in module.rust_source
    warp_signature = module.rust_source.split("async fn warp_main(", 1)[1].split(
        ") -> Result<(), EngineError>", 1
    )[0]
    assert "services: KernelRuntimeServices" in warp_signature
    assert "mbarriers: Arc<PhysicalBarrierHub>" not in warp_signature
    assert "run_kernel_launch_ordered(" in module.rust_source
    assert ".warp_contexts()" not in module.rust_source
    assert "WarpTask::new" not in module.rust_source
    assert "KernelRuntimeServices::new" not in module.rust_source
    assert "completion_registry" not in module.rust_source
    assert "Executor::" not in module.rust_source
    assert "CtaSumHub::cta_sum_hub(topology)" not in module.rust_source
    assert "CollectiveHub::new(move |inputs" not in module.rust_source
    assert "let mut partials = [0.0_f32; WARP_SIZE]" not in module.rust_source
    assert "let mut run_result = RunResultBuilder::new();" in module.rust_source
    assert "run_result.run_phase(" in module.rust_source
    assert "profile_reset(" not in module.rust_source

    def check_execution_stats(value):
        assert value["task_count"] == 6
        assert value["completed_task_count"] == 6
        assert value["poll_order"] == list(range(6))

    expect_harness_surface(lambda: result.stats, check_execution_stats)
    assert "profile_snapshot(" not in module.rust_source
    assert ".detach(" not in module.rust_source
    assert "NumSim kernel phase" not in module.rust_source
    assert "extract_output_allocations(inputs, allocation_ids.len())?" in module.rust_source
    assert "build_run_result(" in module.rust_source
    assert "let kernel_stats = PyList::empty(py)" not in module.rust_source
    assert "let stats_dict = PyDict::new(py)" not in module.rust_source
    assert "let phase_stats = PyDict::new(py)" not in module.rust_source
    assert "snapshot_allocation_bytes" not in module.rust_source
    assert "for (index, allocation) in allocation_ids" not in module.rust_source
    assert "PyBytes" not in module.rust_source


def test_engine_parallelizes_independent_cluster_domains(tmp_path):
    module = numsim.transpile(no_op_kernel, cache_dir=tmp_path)

    result = numsim.Engine(max_workers=8).run(module, {})

    assert result.stats["task_count"] == 6
    assert result.stats["completed_task_count"] == 6
    assert result.stats["worker_count"] == 2
    assert result.stats["scheduling_domain_count"] == 2
    assert sorted(result.stats["poll_order"]) == list(range(6))


def test_warpgroup_relative_scope_ids_map_native_warp_coordinates(tmp_path):
    output = np.zeros((2, 4, 32, 3), dtype=np.int32)
    expected = np.zeros_like(output)
    for warpgroup in range(2):
        for warp in range(4):
            for lane in range(32):
                expected[warpgroup, warp, lane] = (warpgroup, warp, warp * 32 + lane)

    module = numsim.transpile(warpgroup_scope_coordinates, cache_dir=tmp_path)
    result = numsim.Engine().run(module, {"output": output})

    np.testing.assert_array_equal(result.outputs["output"], expected)
    assert result.stats["task_count"] == 8


def test_empty_scope_id_names_do_not_alias_distinct_varying_builtins(tmp_path):
    payload = json.loads(save_json(lane_and_thread_scope_ids))
    for node in payload["nodes"]:
        if node["type"] == "tir.Var" and node["data"]["name"] in {"lane", "thread"}:
            node["data"]["name"] = ""
    kernel = load_json(json.dumps(payload))

    output = np.zeros((4, 32, 2), dtype=np.int32)
    result = numsim.Engine().run(
        numsim.transpile(kernel, cache_dir=tmp_path),
        {"output": output},
    )

    lane = np.broadcast_to(np.arange(32, dtype=np.int32), (4, 32))
    thread = np.arange(128, dtype=np.int32).reshape(4, 32)
    np.testing.assert_array_equal(result.outputs["output"][..., 0], lane)
    np.testing.assert_array_equal(result.outputs["output"][..., 1], thread)


def test_extent_free_warp_id_uses_flat_native_coordinate(tmp_path):
    output = np.zeros((2, 32), dtype=np.int32)
    expected = np.broadcast_to(np.arange(2, dtype=np.int32)[:, None], output.shape)

    module = numsim.transpile(extent_free_warp_id, cache_dir=tmp_path)
    result = numsim.Engine().run(module, {"output": output})

    np.testing.assert_array_equal(result.outputs["output"], expected)


def test_integer_elect_sync_condition_becomes_lane_mask(tmp_path):
    output = np.zeros(32, dtype=np.int32)
    expected = np.full(32, 2, dtype=np.int32)
    expected[0] = 1

    module = numsim.transpile(elect_sync_integer_branch, cache_dir=tmp_path)
    result = numsim.Engine().run(module, {"output": output})

    np.testing.assert_array_equal(result.outputs["output"], expected)
    assert "integer_condition_mask" in module.rust_source


def test_identical_transpile_is_a_cache_hit(tmp_path):
    first = numsim.transpile(no_op_kernel, cache_dir=tmp_path)
    second = numsim.transpile(no_op_kernel, cache_dir=tmp_path)

    assert first.cache_key == second.cache_key
    assert first.library_path == second.library_path
    assert first.load() is second.load()


def test_artifact_cache_identity_ignores_spans_and_rebinds_current_location(tmp_path):
    first = numsim.transpile(no_op_kernel, cache_dir=tmp_path)
    payload = json.loads(save_json(no_op_kernel))
    for node in payload["nodes"]:
        if node["type"] == "ir.SourceName":
            node["data"] = "/tmp/relocated/no_op_kernel.py"
        elif node["type"] == "ir.Span":
            node["data"]["line"] += 2000
            node["data"]["end_line"] += 2000
    relocated = load_json(json.dumps(payload))

    second = numsim.transpile(relocated, cache_dir=tmp_path)

    assert first.cache_key == second.cache_key
    assert first.library_path == second.library_path
    assert first.spec.to_manifest(include_source_spans=False) == second.spec.to_manifest(include_source_spans=False)
    leaves = [
        leaf
        for entry in second.spec.kernels[0].source_map
        for leaf in flatten_source_span(entry.span)
    ]
    assert leaves
    assert {leaf.source_name for leaf in leaves} == {"/tmp/relocated/no_op_kernel.py"}
    assert min(leaf.line for leaf in leaves) >= 2001


def test_single_flat_scope_id_ignores_bound_dynamic_launch_extent(tmp_path):
    output = np.zeros(2, dtype=np.int32)

    module = numsim.transpile(bound_dynamic_cta_extent, cache_dir=tmp_path)
    result = numsim.Engine().run(module, {"output": output})

    np.testing.assert_array_equal(result.outputs["output"], np.array([1, 2], dtype=np.int32))


def test_lane_add_uses_native_masked_warp_control(tmp_path):
    left = np.arange(100, dtype=np.float32)
    right = np.linspace(0, 1, 100, dtype=np.float32)
    output = np.zeros(100, dtype=np.float32)

    module = numsim.transpile(lane_add, cache_dir=tmp_path)
    result = numsim.Engine().run(module, {"left": left, "right": right, "output": output})

    np.testing.assert_array_equal(result.outputs["output"], left + right)
    assert result.stats["task_count"] == 4
    assert "async fn warp_main" in module.rust_source
    assert "parent_mask_branch" in module.rust_source
    assert "WarpValue::from_fn" in module.rust_source
    assert "program counter" not in module.rust_source

    native_output = np.zeros_like(output)
    prepared = prepare_bindings(
        {
            "left": left,
            "right": right,
            "output": native_output,
        }
    )
    native_payload = module.load().run(prepared.to_payload(returned_names={"output"}), None, 1)
    allocation_bytes = native_payload["allocation_bytes"]
    np.testing.assert_array_equal(native_output, left + right)
    assert allocation_bytes[prepared.buffers["left"].allocation] is None
    assert allocation_bytes[prepared.buffers["right"].allocation] is None
    assert allocation_bytes[prepared.buffers["output"].allocation] is None
    assert all(value is None for value in allocation_bytes)


def test_divergent_break_and_continue_use_loop_masks(tmp_path):
    output = np.zeros(128, dtype=np.float32)
    expected = np.zeros_like(output)
    live = np.ones(32, dtype=np.bool_)
    lanes = np.arange(32)
    for step in range(4):
        active = live.copy()
        active[lanes == step] = False
        breaking = active & (lanes < 4) & (step == 2)
        live[breaking] = False
        active[breaking] = False
        expected[lanes[active] * 4 + step] = 1

    module = numsim.transpile(divergent_loop_control, cache_dir=tmp_path)
    result = numsim.Engine().run(module, {"output": output})

    np.testing.assert_array_equal(result.outputs["output"], expected)
    assert result.stats["task_count"] == 1
    assert "live_mask_loop" in module.rust_source
    assert "ctx.set_active_mask(WarpMask::EMPTY)" in module.rust_source


def test_divergent_while_preserves_lane_mask_and_scheduler_observation(tmp_path):
    output = np.zeros(32, dtype=np.float32)
    expected = np.zeros_like(output)
    expected[:16] = 2

    module = numsim.transpile(divergent_while_control, cache_dir=tmp_path)
    result = numsim.Engine().run(module, {"output": output})

    np.testing.assert_array_equal(result.outputs["output"], expected)
    assert result.stats["poll_order"] == [0]


def test_empty_mask_after_continue_skips_mbarrier_wait(tmp_path):
    output = np.zeros(32, dtype=np.int32)

    module = numsim.transpile(mbarrier_wait_after_full_warp_continue, cache_dir=tmp_path)
    result = numsim.Engine().run(module, {"output": output})

    np.testing.assert_array_equal(result.outputs["output"], output)
    assert "if !ctx.active_mask().is_empty()" in module.rust_source


def test_empty_mask_after_continue_skips_named_barrier_argument_evaluation(tmp_path):
    output = np.zeros(32, dtype=np.int32)

    module = numsim.transpile(bar_sync_after_full_warp_continue, cache_dir=tmp_path)
    result = numsim.Engine().run(module, {"output": output})

    np.testing.assert_array_equal(result.outputs["output"], output)
    assert result.stats["task_count"] == 1


def test_empty_mask_after_continue_skips_cta_sync(tmp_path):
    output = np.zeros(32, dtype=np.int32)

    module = numsim.transpile(cta_sync_after_full_warp_continue, cache_dir=tmp_path)
    result = numsim.Engine().run(module, {"output": output})

    np.testing.assert_array_equal(result.outputs["output"], output)
    assert "if !ctx.active_mask().is_empty()" in module.rust_source


def test_empty_mask_after_continue_skips_dynamic_for_argument_evaluation(tmp_path):
    output = np.zeros(32, dtype=np.int32)

    module = numsim.transpile(dynamic_for_after_full_warp_continue, cache_dir=tmp_path)
    result = numsim.Engine().run(module, {"output": output})

    np.testing.assert_array_equal(result.outputs["output"], output)
    assert "if !parent_mask_loop" in module.rust_source


def test_scalar_expression_lowering_matches_floor_and_select_semantics(tmp_path):
    output = np.zeros(32, dtype=np.float32)
    lanes = np.arange(32, dtype=np.int64)
    shifted = lanes - 17
    quotient = shifted // 5
    remainder = shifted % 5
    clamped = np.minimum(np.maximum(quotient, -2), 2)
    use_clamped = ((lanes < 5) & ~(lanes == 2)) | (lanes >= 29)
    expected = np.where(use_clamped, clamped, remainder).astype(np.float32)

    module = numsim.transpile(scalar_expression_mix, cache_dir=tmp_path)
    result = numsim.Engine().run(module, {"output": output})

    np.testing.assert_array_equal(result.outputs["output"], expected)
    assert "floor_div_i64" in module.rust_source
    assert "floor_mod_i64" in module.rust_source
    assert "select_" in module.rust_source


def test_composite_masks_cannot_reactivate_lanes_outside_parent_scope(tmp_path):
    output = np.zeros(32, dtype=np.int32)
    expected = np.zeros_like(output)
    expected[:8] = 2
    expected[8:16] = 4

    module = numsim.transpile(nested_mask_parent_scope, cache_dir=tmp_path)
    result = numsim.Engine().run(module, {"output": output})

    np.testing.assert_array_equal(result.outputs["output"], expected)
    assert "parent_mask_branch_" in module.rust_source
    assert "WarpMask::EMPTY" in module.rust_source


def test_typed_raw_scalar_calls_execute_with_warp_semantics(tmp_path):
    source = 1.0 + np.arange(32, dtype=np.float32) / np.float32(32)
    output = np.zeros(32, dtype=np.float32)
    lanes = np.arange(32, dtype=np.uint32)
    low_nibble = lanes & np.uint32(15)
    shifted = (low_nibble << np.uint32(1)) | (low_nibble >> np.uint32(1))
    mixed = shifted ^ np.uint32(3)
    inverted = (~low_nibble) & np.uint32(15)
    selected = source.copy()
    selected[[0, 31]] = selected[[0, 31]] * np.float32(2) + np.float32(1)
    elected = np.zeros(32, dtype=np.float32)
    elected[0] = 1
    expected = (
        np.exp2(np.log(selected)).astype(np.float32)
        + np.float32(1) / np.sqrt(selected + np.float32(4))
        + (mixed + inverted).astype(np.float32) * np.float32(0.001)
        + elected * np.float32(0.01)
    )

    module = numsim.transpile(raw_scalar_call_mix, cache_dir=tmp_path)
    result = numsim.Engine().run(module, {"source": source, "output": output})

    np.testing.assert_allclose(result.outputs["output"], expected, rtol=2e-06, atol=2e-06)


def test_mbarrier_phase_remains_lane_vector_inside_the_engine(tmp_path):
    output = np.zeros(2, dtype=np.int32)

    module = numsim.transpile(mbarrier_varying_uniform_phase, cache_dir=tmp_path)
    result = numsim.Engine().run(module, {"output": output})

    np.testing.assert_array_equal(result.outputs["output"], np.array([0, 1], dtype=np.int32))


def test_if_then_else_does_not_evaluate_an_unselected_buffer_load(tmp_path):
    source = np.array([3.5], dtype=np.float32)
    output = np.zeros(32, dtype=np.float32)
    expected = np.full(32, 7, dtype=np.float32)
    expected[0] = source[0]

    module = numsim.transpile(guarded_if_then_else_load, cache_dir=tmp_path)
    result = numsim.Engine().run(module, {"source": source, "output": output})

    np.testing.assert_array_equal(result.outputs["output"], expected)
    assert "ctx.set_active_mask(if_then_mask" in module.rust_source


def test_logical_and_or_only_evaluate_rhs_for_required_active_lanes(tmp_path):
    module = numsim.transpile(lazy_logical_buffer_load, cache_dir=tmp_path)
    result = numsim.Engine().run(
        module, {"source": np.array([5], dtype=np.int32), "output": np.zeros(32, dtype=np.int32)}
    )

    expected = np.full(32, 2, dtype=np.int32)
    expected[0] = 3
    np.testing.assert_array_equal(result.outputs["output"], expected)
    assert "ctx.set_active_mask(and_rhs_mask_" in module.rust_source
    assert "ctx.set_active_mask(or_rhs_mask_" in module.rust_source


def test_select_does_not_evaluate_an_unselected_nested_buffer_load(tmp_path):
    source = np.array([17, 23], dtype=np.uint64)
    output = np.zeros(32, dtype=np.uint32)
    expected = np.arange(100, 132, dtype=np.uint32)
    expected[:2] = source.astype(np.uint32)

    module = numsim.transpile(guarded_select_load, cache_dir=tmp_path)
    result = numsim.Engine().run(module, {"source": source, "output": output})

    np.testing.assert_array_equal(result.outputs["output"], expected)
    assert "ctx.set_active_mask(select_then_mask" in module.rust_source
    assert "ctx.set_active_mask(select_else_mask" in module.rust_source


def test_physical_numpy_aliases_survive_the_artifact_boundary(tmp_path, expect_harness_surface):
    backing = np.arange(48, dtype=np.float32)
    original = backing.copy()
    source = backing[:32]
    destination = backing[16:]

    module = numsim.transpile(overlapping_alias_write, cache_dir=tmp_path)
    result = numsim.Engine().run(
        module, {"source": source, "destination": destination}, outputs=("destination",)
    )

    np.testing.assert_array_equal(backing[:16], original[:16])
    np.testing.assert_array_equal(backing[16:], original[:32] + 1)
    np.testing.assert_array_equal(result.outputs["destination"], original[:32] + 1)

    def check_aliasing(value):
        current, expected = value
        np.testing.assert_array_equal(current[16:], expected[:32] + 1)

    expect_harness_surface(
        lambda: (backing, original),
        check_aliasing,
    )


def test_parameter_names_and_explicit_outputs_are_respected(tmp_path):
    left = np.arange(100, dtype=np.float32)
    right = np.ones(100, dtype=np.float32)
    output = np.zeros(100, dtype=np.float32)

    module = numsim.transpile(lane_add, cache_dir=tmp_path)
    result = numsim.Engine().run(
        module,
        {"left": left, "right": right, "output": output},
        outputs=("output",),
    )

    assert set(result.outputs) == {"output"}
    np.testing.assert_array_equal(result.outputs["output"], left + right)


def test_execution_subset_filters_concrete_warp_tasks(tmp_path):
    module = numsim.transpile(no_op_kernel, cache_dir=tmp_path)

    cluster = numsim.Engine().run(module, {}, subset=ExecutionSubset(cluster_ids=[0]))
    cta = numsim.Engine().run(module, {}, subset=ExecutionSubset(cta_ids=[1]))

    assert cluster.stats["task_count"] == 3
    assert cta.stats["task_count"] == 3


def test_execution_subset_rejects_out_of_range_ids(tmp_path):
    module = numsim.transpile(no_op_kernel, cache_dir=tmp_path)

    with pytest.raises(numsim.NumSimExecutionError, match="outside"):
        numsim.Engine().run(module, {}, subset=ExecutionSubset(cluster_ids=[2]))


def test_run_case_uses_declared_output_bindings(monkeypatch, tmp_path):
    monkeypatch.setenv("NUMSIM_CACHE_DIR", str(tmp_path))
    left = np.arange(100, dtype=np.float32)
    right = np.linspace(0, 1, 100, dtype=np.float32)
    output = np.zeros(100, dtype=np.float32)
    expected = left + right
    case = numsim.NumSimCase(
        kernel=lane_add,
        args={"left": left, "right": right, "output": output},
        outputs=("output",),
        reference=lambda: {"output": expected.copy()},
    )

    report = numsim.run_case(case)

    assert report.ok
    np.testing.assert_array_equal(output, expected)


def test_multidimensional_default_layout_lowers_to_physical_offsets(tmp_path):
    left = np.arange(15, dtype=np.float32).reshape(3, 5)
    right = np.linspace(0, 1, 15, dtype=np.float32).reshape(3, 5)
    output = np.zeros((3, 5), dtype=np.float32)

    module = numsim.transpile(matrix_add_2d, cache_dir=tmp_path)
    result = numsim.Engine().run(module, {"left": left, "right": right, "output": output})

    np.testing.assert_array_equal(result.outputs["output"], left + right)
    assert "T.TileLayout" not in module.rust_source


def test_shared_allocations_alias_and_are_isolated_per_cta(tmp_path):
    output = np.zeros((2, 32), dtype=np.float32)
    expected = np.stack([np.arange(32, dtype=np.float32), 100 + np.arange(32, dtype=np.float32)])

    module = numsim.transpile(shared_alias_per_cta, cache_dir=tmp_path)
    result = numsim.Engine().run(module, {"output": output})

    np.testing.assert_array_equal(result.outputs["output"], expected)
    assert "runtime_buffer_shared(" in module.rust_source


def test_shared_uninitialized_read_is_zero_filled_and_requires_review(tmp_path):
    module = numsim.transpile(shared_uninitialized_read, cache_dir=tmp_path)
    output = np.full(32, np.float32(7), dtype=np.float32)

    result = numsim.Engine().run(module, {"output": output})

    np.testing.assert_array_equal(result.outputs["output"], np.zeros(32, dtype=np.float32))
    assert result.verdict == "review"
    assert result.diagnostics
    assert {item["status"] for item in result.diagnostics} == {"review"}
    assert {item["kind"] for item in result.diagnostics} == {"uninitialized_read"}

    report = numsim.compare(result, {"output": np.ones(32, dtype=np.float32)})
    assert report.verdict == "error"
    assert not report.ok


def test_local_register_arrays_are_isolated_per_lane(tmp_path):
    output = np.zeros(32, dtype=np.float32)
    expected = 2 * np.arange(32, dtype=np.float32) + 1

    module = numsim.transpile(local_array_per_lane, cache_dir=tmp_path)
    result = numsim.Engine().run(module, {"output": output})

    np.testing.assert_array_equal(result.outputs["output"], expected)
    assert "runtime_buffer_register(" in module.rust_source


def test_scalar_buffer_dtypes_use_physical_little_endian_storage(tmp_path):
    input_i32 = np.arange(-16, 16, dtype=np.int32)
    output_i32 = np.zeros(32, dtype=np.int32)
    input_u64 = np.arange(32, dtype=np.uint64) * np.uint64(1000)
    output_u64 = np.zeros(32, dtype=np.uint64)
    input_f16 = np.linspace(-4, 4, 32, dtype=np.float16)
    output_f16 = np.zeros(32, dtype=np.float16)
    input_bf16_bits = _encode_bf16(np.linspace(-3, 3, 32, dtype=np.float32))
    output_bf16_bits = np.zeros(32, dtype=np.uint16)
    input_bool = np.arange(32) % 3 == 0
    output_bool = np.zeros(32, dtype=np.bool_)

    input_bf16 = input_bf16_bits
    output_bf16 = output_bf16_bits
    module = numsim.transpile(scalar_buffer_types, cache_dir=tmp_path)
    result = numsim.Engine().run(
        module,
        {
            "input_i32": input_i32,
            "output_i32": output_i32,
            "input_u64": input_u64,
            "output_u64": output_u64,
            "input_f16": input_f16,
            "output_f16": output_f16,
            "input_bf16": input_bf16,
            "output_bf16": output_bf16,
            "input_bool": input_bool,
            "output_bool": output_bool,
        },
    )

    np.testing.assert_array_equal(result.outputs["output_i32"], input_i32 + 7)
    np.testing.assert_array_equal(result.outputs["output_u64"], input_u64 + 11)
    expected_f16 = (input_f16.astype(np.float32) + 0.5).astype(np.float16)
    np.testing.assert_array_equal(result.outputs["output_f16"], expected_f16)
    expected_bf16 = _encode_bf16(_decode_bf16(input_bf16_bits) + 0.5)
    np.testing.assert_array_equal(result.outputs["output_bf16"], expected_bf16)
    np.testing.assert_array_equal(result.outputs["output_bool"], input_bool)


def test_runtime_shape_scalars_are_bound_from_consistent_buffer_descriptors(tmp_path):
    source = np.arange(5 * 32, dtype=np.float32).reshape(5, 32)
    output = np.zeros_like(source)

    module = numsim.transpile(dynamic_rows, cache_dir=tmp_path)
    result = numsim.Engine().run(
        module,
        {"input_buffer": source, "output_buffer": output},
        outputs=("output_buffer",),
    )

    np.testing.assert_array_equal(result.outputs["output_buffer"], source + 2)
    assert "extract_shape_extent" in module.rust_source
    assert "fn extract_shape_extent(" not in module.rust_source


def test_runtime_shape_scalars_reject_disagreeing_bindings(tmp_path):
    module = numsim.transpile(dynamic_rows, cache_dir=tmp_path)

    with pytest.raises(numsim.NumSimExecutionError, match="runtime shape 'rows' disagrees"):
        numsim.Engine().run(
            module,
            {
                "input_buffer": np.zeros((5, 32), dtype=np.float32),
                "output_buffer": np.zeros((4, 32), dtype=np.float32),
            },
        )


def test_compose_swizzle_layout_resolves_to_physical_alias_bytes(tmp_path):
    output = np.zeros(32, dtype=np.float32)

    module = numsim.transpile(compose_swizzle_alias, cache_dir=tmp_path)
    result = numsim.Engine().run(module, {"output": output})

    np.testing.assert_array_equal(result.outputs["output"], np.arange(1, 33, dtype=np.float32))


def test_padded_compose_layout_allocates_its_physical_span_for_aliases(tmp_path):
    output = np.zeros(6, dtype=np.uint32)

    module = numsim.transpile(padded_compose_alias, cache_dir=tmp_path)
    result = numsim.Engine().run(module, {"output": output})

    np.testing.assert_array_equal(
        result.outputs["output"], np.array([10, 11, 90, 91, 12, 13], dtype=np.uint32)
    )


def test_cta_reductions_preserve_each_warp_partial_in_scratch(tmp_path):
    output = np.zeros((3, 3), dtype=np.float32)

    module = numsim.transpile(cta_reductions_preserve_scratch, cache_dir=tmp_path)
    result = numsim.Engine().run(module, {"output": output})

    np.testing.assert_array_equal(
        result.outputs["output"],
        np.array(
            [[4192.0, 4192.0, 3696.0], [131.0, 131.0, 131.0], [0.0, 0.0, 100.0]],
            dtype=np.float32,
        ),
    )


def test_cuda_float_reductions_match_nan_and_signed_zero_bits(tmp_path):
    operands = np.array([0x80000000, 0x00000000], dtype=np.uint32)
    output = np.zeros(7, dtype=np.uint32)

    module = numsim.transpile(cuda_float_reduction_edge_bits, cache_dir=tmp_path)
    result = numsim.Engine().run(module, {"operands": operands, "output": output})

    np.testing.assert_array_equal(
        result.outputs["output"],
        np.array(
            [0x7FFFFFFF, 0x00000000, 0x80000000, 0x00000000, 0x00000000, 0x80000000, 0x80000000],
            dtype=np.uint32,
        ),
    )
    assert "cuda_f32_max" in module.rust_source
    assert "cuda_f32_min" in module.rust_source


def test_address_of_lowers_to_integer_bits_before_program_use(tmp_path):
    module = numsim.transpile(physical_address_value, cache_dir=tmp_path)
    result = numsim.Engine().run(module, {})
    metadata = module.load().metadata()

    bool(result.stats["completed_task_count"] == 1)
    for name, expected in abi_metadata().items():
        assert metadata[name] == expected
    assert metadata["engine_hash"] == module.artifact.manifest["engine_hash"]
    assert metadata["build_identity"] == module.artifact.manifest["build_identity"]
    assert metadata["engine_hash"] == metadata["build_identity"]["engine_hash"]
    assert "use numsim_engine::artifact_support::*;" in module.rust_source
    assert "import_generated_surface" not in module.rust_source
    assert "numsim_engine::abi::staging" not in module.rust_source
    assert "use numsim_engine::abi::staging::*;" not in module.rust_source
    assert "use numsim_engine::runtime::*;" not in module.rust_source
    assert "struct PhysicalPtr" not in module.rust_source
    assert "PhysicalPtr::new" in module.rust_source
    assert "generic_addresses_u64" in module.rust_source
    assert "tirx.address_of" not in module.rust_source


def test_warp_local_raw_calls_execute_inside_one_warp_unit(tmp_path):
    source = np.linspace(-2, 2, 32, dtype=np.float32)
    output = np.zeros(32, dtype=np.float32)
    reduced = np.zeros(32, dtype=np.uint32)
    packed = np.zeros(32, dtype=np.uint32)

    module = numsim.transpile(warp_pure_calls, cache_dir=tmp_path)
    result = numsim.Engine().run(
        module, {"source": source, "output": output, "reduced": reduced, "packed": packed}
    )

    reverse = source[::-1].copy()
    expected = reverse + np.float32(1) / (np.abs(source) + np.float32(1)) + 3 * source
    np.testing.assert_allclose(result.outputs["output"], expected, rtol=1e-06, atol=1e-06)
    np.testing.assert_array_equal(result.outputs["reduced"], np.full(32, 497, dtype=np.uint32))
    low = _encode_bf16(source).astype(np.uint32)
    high = _encode_bf16(reverse).astype(np.uint32) << np.uint32(16)
    np.testing.assert_array_equal(result.outputs["packed"], low | high)
    assert result.stats["task_count"] == 1


def test_native_varying_assert_reports_failed_lane_mask(tmp_path):
    module = numsim.transpile(native_varying_assert, cache_dir=tmp_path)

    with pytest.raises(numsim.NumSimExecutionError, match="failed lanes=.*31"):
        numsim.Engine().run(module, {"output": np.zeros(32, dtype=np.float32)})


def test_dps_float_arithmetic_writes_through_lane_private_physical_pointers(tmp_path):
    output = np.zeros((32, 5), dtype=np.float32)

    module = numsim.transpile(dps_float_arithmetic, cache_dir=tmp_path)
    result = numsim.Engine().run(module, {"output": output})

    lane = np.arange(32, dtype=np.float32)
    expected = np.stack(
        [lane * 2 + 1, lane * 2 + 4, (lane + 1) * 3 + 5, lane + 2, lane + 4], axis=1
    )
    np.testing.assert_array_equal(result.outputs["output"], expected)


def test_dps_writes_through_the_full_physical_pointer(tmp_path):
    output = np.full(33, -1, dtype=np.float32)
    module = numsim.transpile(dps_pointer_offset, cache_dir=tmp_path)

    result = numsim.Engine().run(module, {"output": output})

    expected = np.concatenate(
        (np.array([-1], dtype=np.float32), np.arange(100, 132, dtype=np.float32))
    )
    np.testing.assert_array_equal(result.outputs["output"], expected)


@pytest.mark.parametrize(
    ("kernel", "message"),
    [
        (dps_read_only_destination, "non-writable physical pointer"),
    ],
)
def test_dps_preserves_pointer_access_contracts(kernel, message, tmp_path):
    module = numsim.transpile(kernel, cache_dir=tmp_path)

    with pytest.raises(numsim.NumSimExecutionError, match=message):
        numsim.Engine().run(module, {"output": np.zeros(33, dtype=np.float32)})


def test_dps_integer_pointer_arithmetic_does_not_carry_access_ptr_extent(tmp_path):
    module = numsim.transpile(dps_destination_after_access_extent, cache_dir=tmp_path)
    result = numsim.Engine().run(module, {"output": np.zeros(33, dtype=np.float32)})

    expected = np.concatenate(
        (np.zeros(1, dtype=np.float32), np.arange(100, 132, dtype=np.float32))
    )
    np.testing.assert_array_equal(result.outputs["output"], expected)


def test_physical_mbarrier_reuses_phase_and_completes_numeric_transactions_directly(tmp_path):
    source = np.arange(4, dtype=np.float32)
    output = np.zeros(2, dtype=np.int32)

    module = numsim.transpile(mbarrier_phase_reuse, cache_dir=tmp_path)
    result = numsim.Engine().run(module, {"source": source, "output": output})

    np.testing.assert_array_equal(result.outputs["output"], np.array([1, 2], dtype=np.int32))
    assert result.stats["completed_task_count"] == 2
    assert result.stats["completion_operation_count"] == 0
    assert result.stats["completion_pump_count"] >= 1


def test_physical_mbarrier_deadlock_reports_missing_lane_arrivals(tmp_path, expect_harness_error):
    module = numsim.transpile(mbarrier_missing_arrivals, cache_dir=tmp_path)

    def action():
        return numsim.Engine(max_workers=1).run(module, {})

    def check_error_payload(error):
        assert "arrival_count=32/64" in str(error)

    def check_deadlock(error):
        assert "deadlock" in str(error).lower()

    expect_harness_error(
        action,
        error=numsim.NumSimExecutionError,
        match="arrival_count=32/64",
        check=check_error_payload,
    )
    expect_harness_error(
        action,
        error=numsim.NumSimExecutionError,
        match="arrival_count=32/64",
        check=check_deadlock,
    )


def test_remote_mbarrier_arrive_targets_the_other_ctas_physical_slot(tmp_path):
    output = np.zeros(2, dtype=np.int32)

    module = numsim.transpile(mbarrier_remote_cta, cache_dir=tmp_path)
    result = numsim.Engine().run(module, {"output": output})

    np.testing.assert_array_equal(result.outputs["output"], np.ones(2, dtype=np.int32))


def test_remote_mbarrier_explicit_count_completes_the_target_phase(tmp_path):
    output = np.zeros(2, dtype=np.int32)

    module = numsim.transpile(remote_mbarrier_explicit_count, cache_dir=tmp_path)
    result = numsim.Engine().run(module, {"output": output})

    np.testing.assert_array_equal(result.outputs["output"], np.ones(2, dtype=np.int32))


def test_cluster_mbarrier_arrive_recovers_target_from_pointer_derived_view(tmp_path):
    output = np.zeros(2, dtype=np.int32)

    module = numsim.transpile(mapped_remote_mbarrier_cluster_view, cache_dir=tmp_path)
    result = numsim.Engine().run(module, {"output": output})

    np.testing.assert_array_equal(result.outputs["output"], np.ones(2, dtype=np.int32))


def test_mapa_remote_shared_reads_and_subgroup_reductions_run_inside_one_warp(tmp_path):
    output = np.zeros((2, 3, 32), dtype=np.float32)
    expected = np.zeros_like(output)
    expected[:, 0, :2] = 3
    expected[:, 1, :2] = 2
    expected[:, 2, :2] = 1

    module = numsim.transpile(remote_shared_read_and_warp_reduce, cache_dir=tmp_path)
    result = numsim.Engine().run(module, {"output": output})

    np.testing.assert_array_equal(result.outputs["output"], expected)


def test_mapa_remote_shared_writes_use_the_target_ctas_owner(tmp_path):
    output = np.zeros(2, dtype=np.float32)

    module = numsim.transpile(remote_shared_write_ownership, cache_dir=tmp_path)
    result = numsim.Engine(max_workers=8).run(module, {"output": output})

    np.testing.assert_array_equal(result.outputs["output"], np.array([10, 11], dtype=np.float32))
    assert result.stats["worker_count"] == 1
    assert result.stats["scheduling_domain_count"] == 1


def test_bulk_shared_to_cluster_consumes_public_u64_mapa_addresses(tmp_path):
    module = numsim.transpile(bulk_shared_to_cluster_u64_addresses, cache_dir=tmp_path)
    result = numsim.Engine().run(module, {"output": np.zeros((2, 4), dtype=np.float32)})

    expected = np.array([[1, 2, 3, 4], [1, 2, 3, 4]], dtype=np.float32)
    np.testing.assert_array_equal(result.outputs["output"], expected)


def test_bulk_shared_to_cluster_has_exact_racecheck_payload_accesses():
    report = racecheck(
        bulk_shared_to_cluster_u64_addresses,
        inputs={"output": np.zeros((2, 4), dtype=np.float32)},
    )

    assert report.verdict == "clean", report.format()


def test_parallel_clusters_keep_remote_shared_ownership_isolated(tmp_path):
    output = np.zeros((2, 2, 2), dtype=np.float32)
    expected = np.array([[[1, 11], [11, 1]], [[101, 111], [111, 101]]], dtype=np.float32)

    module = numsim.transpile(parallel_cluster_remote_shared_exchange, cache_dir=tmp_path)
    result = numsim.Engine(max_workers=2).run(module, {"output": output})

    np.testing.assert_array_equal(result.outputs["output"], expected)
    assert result.stats["task_count"] == 4
    assert result.stats["completed_task_count"] == 4
    assert result.stats["worker_count"] == 2
    assert result.stats["scheduling_domain_count"] == 2


def test_local_mbarrier_arrive_rejects_mapped_remote_address(tmp_path):
    output = np.zeros(2, dtype=np.int32)

    module = numsim.transpile(mapped_remote_mbarrier_pointer, cache_dir=tmp_path)
    with pytest.raises(
        numsim.NumSimExecutionError,
        match=r"local-form mbarrier\.arrive.*global CTA 1.*remote global CTA 0",
    ):
        numsim.Engine().run(module, {"output": output})


def test_local_mbarrier_arrive_expect_tx_rejects_mapped_remote_address(tmp_path):
    output = np.zeros(2, dtype=np.int32)

    module = numsim.transpile(mapped_remote_mbarrier_pointer_expect_tx, cache_dir=tmp_path)
    with pytest.raises(
        numsim.NumSimExecutionError,
        match=r"local-form mbarrier\.arrive.*global CTA 1.*remote global CTA 0",
    ):
        numsim.Engine().run(module, {"output": output})


def test_scoped_and_named_barriers_rendezvous_across_warps_and_ctas(tmp_path):
    output = np.zeros((2, 4, 32), dtype=np.int32)
    cta = np.arange(2, dtype=np.int32)[:, None, None]
    warp = np.arange(4, dtype=np.int32)[None, :, None]
    lane = np.arange(32, dtype=np.int32)[None, None, :]
    expected = cta * 10000 + warp * 100 + lane

    module = numsim.transpile(scoped_syncs, cache_dir=tmp_path)
    result = numsim.Engine().run(module, {"output": output})

    np.testing.assert_array_equal(result.outputs["output"], expected)
    assert result.stats["completed_task_count"] == 8


def test_cluster_rendezvous_rejects_partial_cluster_subset(tmp_path):
    output = np.full((2, 4, 32), -1, dtype=np.int32)

    module = numsim.transpile(scoped_syncs, cache_dir=tmp_path)
    with pytest.raises(
        numsim.NumSimExecutionError, match="CTA subset must be a union of complete clusters"
    ):
        numsim.Engine().run(module, {"output": output}, subset=ExecutionSubset(cta_ids=[0]))


def test_cluster_rendezvous_accepts_complete_cluster_selections(tmp_path):
    cta = np.arange(2, dtype=np.int32)[:, None, None]
    warp = np.arange(4, dtype=np.int32)[None, :, None]
    lane = np.arange(32, dtype=np.int32)[None, None, :]
    expected = cta * 10000 + warp * 100 + lane

    module = numsim.transpile(scoped_syncs, cache_dir=tmp_path)
    for subset in (ExecutionSubset(cluster_ids=[0]), ExecutionSubset(cta_ids=[0, 1])):
        output = np.full((2, 4, 32), -1, dtype=np.int32)
        result = numsim.Engine().run(module, {"output": output}, subset=subset)

        np.testing.assert_array_equal(result.outputs["output"], expected)
        assert result.stats["task_count"] == 8
