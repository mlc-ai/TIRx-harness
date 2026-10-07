from __future__ import annotations

import numpy as np
import pytest
import tvm

from tirx_harness import numsim
from tests.numsim.support.kernels import ordering_only_control_calls
from tests.numsim.support.manifest import emitted_calls
from tvm.script import tirx as T


@T.prim_func
def setmaxnreg_ordering_only(output: T.Buffer((1,), "int32")):
    T.device_entry()
    warp = T.warp_id([4])
    lane = T.lane_id([32])
    T.ptx.setmaxnreg.inc.sync.aligned.u32(24)
    T.cuda.warpgroup_sync(7)
    T.ptx.setmaxnreg.dec.sync.aligned.u32(256)
    if (warp == 0) and (lane == 0):
        output[0] = 7


@T.prim_func
def setmaxnreg_without_intervening_sync(output: T.Buffer((1,), "int32")):
    T.device_entry()
    warp = T.warp_id([4])
    lane = T.lane_id([32])
    T.ptx.setmaxnreg.inc.sync.aligned.u32(24)
    T.ptx.setmaxnreg.dec.sync.aligned.u32(256)
    if (warp == 0) and (lane == 0):
        output[0] = 7


@T.prim_func
def setmaxnreg_warp_disagreement(output: T.Buffer((1,), "int32")):
    T.device_entry()
    warp = T.warp_id([4])
    lane = T.lane_id([32])
    if warp == 0:
        T.ptx.setmaxnreg.inc.sync.aligned.u32(24)
    else:
        T.ptx.setmaxnreg.inc.sync.aligned.u32(32)
    if (warp == 0) and (lane == 0):
        output[0] = 7


@T.prim_func
def setmaxnreg_equivalent_branch_sites(output: T.Buffer((1,), "int32")):
    T.device_entry()
    warp = T.warp_id([4])
    lane = T.lane_id([32])
    if warp == 0:
        T.ptx.setmaxnreg.dec.sync.aligned.u32(24)
    elif warp == 1:
        T.ptx.setmaxnreg.dec.sync.aligned.u32(24)
    elif warp == 2:
        T.ptx.setmaxnreg.dec.sync.aligned.u32(24)
    else:
        T.ptx.setmaxnreg.dec.sync.aligned.u32(24)
    if (warp == 0) and (lane == 0):
        output[0] = 11


@T.prim_func
def setmaxnreg_deleted(output: T.Buffer((1,), "int32")):
    T.device_entry()
    warp = T.warp_id([4])
    lane = T.lane_id([32])
    T.cuda.warpgroup_sync(7)
    if (warp == 0) and (lane == 0):
        output[0] = 7


def _setmaxnreg_static_expression_kernel():
    return tvm.script.from_source(
        """
@T.prim_func
def setmaxnreg_static_expressions(output: T.Buffer((1,), "int32")):
    T.device_entry()
    warp = T.warp_id([4])
    lane = T.lane_id([32])
    T.ptx.setmaxnreg.inc.sync.aligned.u32(T.int32(24) + T.int32(8))
    T.cuda.warpgroup_sync(7)
    T.ptx.setmaxnreg.dec.sync.aligned.u32(T.int32(72) - T.int32(8))
    if (warp == 0) and (lane == 0):
        output[0] = 1
""",
        extra_vars={"T": T},
    )


@T.prim_func
def divergent_warp_sync(output: T.Buffer((32,), "int32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    if lane < 16:
        T.cuda.warp_sync()
    output[lane] = lane


@T.prim_func
def griddep_producer(intermediate: T.Buffer((32,), "int32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    intermediate[lane] = lane + 1
    T.ptx.griddepcontrol.launch_dependents()


@T.prim_func
def griddep_consumer(intermediate: T.Buffer((32,), "int32"), output: T.Buffer((32,), "int32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    T.ptx.griddepcontrol.wait()
    output[lane] = intermediate[lane] * 2


def test_ordering_only_calls_emit_the_registered_instructions():
    assert [
        call.function
        for call in emitted_calls(ordering_only_control_calls, "tirx.ptx.fence_mbarrier_init")
    ] == ["v2::sync::fence_mbarrier_init"]
    assert [
        call.function
        for call in emitted_calls(ordering_only_control_calls, "tirx.ptx.griddepcontrol")
    ] == ["v2::control::griddepcontrol"]


@pytest.mark.parametrize(
    ("statement", "message"),
    (
        (
            "T.ptx.fence.mbarrier_init.release.cluster(T.int32(0))",
            "fence_mbarrier_init expects 0 operand",
        ),
        (
            "T.ptx.griddepcontrol.wait(T.int32(0))",
            "griddepcontrol expects 0 operand",
        ),
    ),
)
def test_target_ptx_parser_rejects_extra_ordering_operands(statement, message):
    source = f"""
@T.prim_func
def invalid():
    T.device_entry()
    {statement}
"""
    with pytest.raises(tvm.error.DiagnosticError, match=message):
        tvm.script.from_source(source, {"T": T})


def test_ordering_only_calls_preserve_native_source_order(tmp_path):
    output = np.zeros(32, dtype=np.int32)

    module = numsim.transpile(ordering_only_control_calls, cache_dir=tmp_path)
    assert module.spec.kernels[0].semantic_requirements == ("external_grid_dependency_satisfied",)
    result = numsim.Engine().run(module, {"output": output})

    np.testing.assert_array_equal(result.outputs["output"], np.arange(1, 33, dtype=np.int32))
    assert "v2::control::griddepcontrol(" in module.rust_source
    assert "tirx.ptx.fence_mbarrier_init" not in module.rust_source
    assert "tirx.ptx.griddepcontrol_wait" not in module.rust_source


def test_setmaxnreg_is_an_ordering_call_not_a_tcgen_lifecycle_call(tmp_path):
    calls = emitted_calls(setmaxnreg_ordering_only, "tirx.ptx.setmaxnreg")
    assert [call.function for call in calls] == ["v2::control::setmaxnreg"] * 2

    module = numsim.transpile(setmaxnreg_ordering_only, cache_dir=tmp_path)
    result = numsim.Engine().run(module, {"output": np.zeros(1, dtype=np.int32)})
    np.testing.assert_array_equal(result.outputs["output"], np.array([7], dtype=np.int32))


def test_setmaxnreg_static_expressions_follow_public_parser_and_runtime(tmp_path):
    module = numsim.transpile(_setmaxnreg_static_expression_kernel(), cache_dir=tmp_path)
    result = numsim.Engine().run(module, {"output": np.zeros(1, dtype=np.int32)})

    np.testing.assert_array_equal(result.outputs["output"], np.array([1], dtype=np.int32))


@pytest.mark.parametrize(
    ("parameters", "register_count", "message"),
    (
        (
            "nreg: T.int32",
            "nreg",
            "compile-time integer constant, got Var",
        ),
        (
            "",
            'T.Cast("int32", T.int64(64))',
            "compile-time integer constant, got Cast",
        ),
        (
            "",
            "16",
            r"must be one of .* got 16",
        ),
        (
            "",
            "25",
            r"must be one of .* got 25",
        ),
        (
            "",
            "264",
            r"must be one of .* got 264",
        ),
    ),
)
def test_setmaxnreg_parser_rejects_nonliteral_and_illegal_counts(
    parameters, register_count, message
):
    source = f"""
@T.prim_func
def invalid({parameters}):
    T.device_entry()
    T.ptx.setmaxnreg.inc.sync.aligned.u32({register_count})
"""
    with pytest.raises(tvm.error.DiagnosticError, match=message):
        tvm.script.from_source(source, {"T": T})


def test_setmaxnreg_requires_explicit_warpgroup_sync_before_a_later_call(tmp_path):
    module = numsim.transpile(setmaxnreg_without_intervening_sync, cache_dir=tmp_path)
    with pytest.raises(numsim.NumSimExecutionError, match="without an explicit warpgroup"):
        numsim.Engine().run(module, {"output": np.zeros(1, dtype=np.int32)})


def test_setmaxnreg_rejects_warp_disagreement_within_one_occurrence(tmp_path):
    module = numsim.transpile(setmaxnreg_warp_disagreement, cache_dir=tmp_path)
    with pytest.raises(numsim.NumSimExecutionError, match="disagreed across warps"):
        numsim.Engine().run(module, {"output": np.zeros(1, dtype=np.int32)})


def test_setmaxnreg_accepts_equivalent_requests_from_distinct_branch_sites(tmp_path):
    module = numsim.transpile(setmaxnreg_equivalent_branch_sites, cache_dir=tmp_path)

    result = numsim.Engine().run(module, {"output": np.zeros(1, dtype=np.int32)})

    np.testing.assert_array_equal(result.outputs["output"], np.array([11], dtype=np.int32))


def test_deleting_setmaxnreg_keeps_the_numerical_result(tmp_path):
    original = numsim.transpile(setmaxnreg_ordering_only, cache_dir=tmp_path / "original")
    deleted = numsim.transpile(setmaxnreg_deleted, cache_dir=tmp_path / "deleted")

    original_result = numsim.Engine().run(original, {"output": np.zeros(1, dtype=np.int32)})
    deleted_result = numsim.Engine().run(deleted, {"output": np.zeros(1, dtype=np.int32)})

    np.testing.assert_array_equal(
        original_result.outputs["output"], deleted_result.outputs["output"]
    )


def test_default_full_mask_warp_sync_rejects_divergent_execution(tmp_path):
    module = numsim.transpile(divergent_warp_sync, cache_dir=tmp_path)

    with pytest.raises(
        numsim.NumSimExecutionError,
        match="participant mask names an inactive lane",
    ):
        numsim.Engine().run(module, {"output": np.zeros(32, dtype=np.int32)})


def test_griddep_token_crosses_sequential_kernel_phases(tmp_path):
    intermediate = np.zeros(32, dtype=np.int32)
    output = np.zeros(32, dtype=np.int32)
    module = numsim.transpile((griddep_producer, griddep_consumer), cache_dir=tmp_path)

    result = numsim.Engine().run(
        module,
        {
            "k0:intermediate": intermediate,
            "k1:intermediate": intermediate,
            "k1:output": output,
        },
        outputs=("k1:output",),
    )

    np.testing.assert_array_equal(result.outputs["k1:output"], 2 * np.arange(1, 33, dtype=np.int32))
    assert "ordering.griddep_" not in module.rust_source
