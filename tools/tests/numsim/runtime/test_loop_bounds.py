from __future__ import annotations

import numpy as np
import pytest

from tirx_harness import numsim
from tvm.script import tirx as T

BT = 16


@T.prim_func
def loop_var_dtype_select(output: T.Buffer((32,), "int32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    for iteration in T.serial(3):
        output[lane] = T.Select(lane < 16, T.int32(7), iteration)


@T.prim_func
def runtime_for_range(
    minimum: T.Buffer((32,), "int32"),
    extents: T.Buffer((32,), "int32"),
    steps: T.Buffer((32,), "int32"),
    output: T.Buffer((32,), "int32"),
    selected: T.Buffer((32,), "int32"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    for iteration in T.serial(minimum[lane], minimum[lane] + extents[lane], step=steps[lane]):
        output[lane] = iteration
        selected[lane] = T.Select(lane < 16, T.int32(7), iteration)


@T.prim_func
def nested_triangular_for(output: T.Buffer((32,), "int32")):
    T.device_entry()
    _warp = T.warp_id([1])
    j = T.lane_id([32])
    output[j] = 0
    if j < BT:
        for outer in T.serial(j + 1, BT):
            output[j] = output[j] + 1
            for _inner in T.serial(j + 1, outer):
                output[j] = output[j] + 1


@T.prim_func
def varying_loop_control(extents: T.Buffer((32,), "int32"), output: T.Buffer((32,), "int32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    output[lane] = 0
    for iteration in T.serial(extents[lane]):
        if iteration == 1:
            if lane % 2 == 0:
                continue
        if iteration == 2:
            if lane % 3 == 0:
                break
        output[lane] = output[lane] + 1


@T.prim_func
def static_zero_trip_for(output: T.Buffer((32,), "int32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    output[lane] = 7
    for iteration in T.serial(0):
        output[lane] = iteration + 100


@T.prim_func
def static_single_trip_for(inside: T.Buffer((32,), "int32"), after: T.Buffer((32,), "int32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    inside[lane] = 0
    for iteration in T.serial(1):
        if lane < 8:
            continue
        inside[lane] = iteration + 1
        if lane < 16:
            break
        inside[lane] = inside[lane] + 2
    after[lane] = 9


def test_uniform_loop_var_preserves_tir_dtype_in_select(tmp_path):
    module = numsim.transpile(loop_var_dtype_select, cache_dir=tmp_path)
    result = numsim.Engine().run(module, {"output": np.zeros(32, dtype=np.int32)})

    expected = np.where(np.arange(32) < 16, 7, 2).astype(np.int32)
    np.testing.assert_array_equal(result.outputs["output"], expected)
    assert ": i32 =" in module.rust_source


def test_lane_varying_min_extent_and_step_use_masked_native_loop(tmp_path):
    minimum = np.arange(32, dtype=np.int32) - 16
    extents = np.arange(32, dtype=np.int32) % 9 + 1
    steps = np.arange(32, dtype=np.int32) % 4 + 1
    initial = np.full(32, -999, dtype=np.int32)

    module = numsim.transpile(runtime_for_range, cache_dir=tmp_path)
    result = numsim.Engine().run(
        module,
        {
            "minimum": minimum,
            "extents": extents,
            "steps": steps,
            "output": initial.copy(),
            "selected": initial.copy(),
        },
    )

    last = minimum + ((extents - 1) // steps) * steps
    selected = np.where(np.arange(32) < 16, 7, last).astype(np.int32)
    np.testing.assert_array_equal(result.outputs["output"], last)
    np.testing.assert_array_equal(result.outputs["selected"], selected)
    assert "iteration_mask_loop" in module.rust_source
    assert "WarpValue<i32>" in module.rust_source
    assert "differs between active lanes" not in module.rust_source

    invalid_steps = steps.copy()
    invalid_steps[5] = 0
    with pytest.raises(numsim.NumSimExecutionError, match="For step must be positive"):
        numsim.Engine().run(
            module,
            {
                "minimum": minimum,
                "extents": extents,
                "steps": invalid_steps,
                "output": initial.copy(),
                "selected": initial.copy(),
            },
        )


def test_v0_style_nested_triangular_bounds(tmp_path):
    module = numsim.transpile(nested_triangular_for, cache_dir=tmp_path)
    result = numsim.Engine().run(module, {"output": np.zeros(32, dtype=np.int32)})

    expected = np.zeros(32, dtype=np.int32)
    outer_iterations = BT - 1 - np.arange(BT, dtype=np.int32)
    expected[:BT] = outer_iterations * (outer_iterations + 1) // 2
    np.testing.assert_array_equal(result.outputs["output"], expected)
    assert module.rust_source.count("iteration_mask_loop") >= 2


def test_lane_varying_loop_supports_break_and_continue(tmp_path):
    extents = np.arange(32, dtype=np.int32) % 6
    expected = np.zeros(32, dtype=np.int32)
    for lane, extent in enumerate(extents):
        for iteration in range(int(extent)):
            if iteration == 1 and lane % 2 == 0:
                continue
            if iteration == 2 and lane % 3 == 0:
                break
            expected[lane] += 1

    module = numsim.transpile(varying_loop_control, cache_dir=tmp_path)
    result = numsim.Engine().run(
        module, {"extents": extents, "output": np.zeros(32, dtype=np.int32)}
    )

    np.testing.assert_array_equal(result.outputs["output"], expected)
    assert "live_mask_loop" in module.rust_source
    assert "ctx.set_active_mask(WarpMask::EMPTY)" in module.rust_source


def test_static_zero_trip_for_omits_native_loop_scaffolding(tmp_path):
    module = numsim.transpile(static_zero_trip_for, cache_dir=tmp_path)
    result = numsim.Engine().run(module, {"output": np.zeros(32, dtype=np.int32)})

    np.testing.assert_array_equal(result.outputs["output"], np.full(32, 7, dtype=np.int32))
    assert "loop_offset_loop" not in module.rust_source
    assert "live_mask_loop" not in module.rust_source


def test_static_single_trip_for_preserves_masks_and_loop_path(tmp_path):
    args = {
        "inside": np.zeros(32, dtype=np.int32),
        "after": np.zeros(32, dtype=np.int32),
    }
    module = numsim.transpile(static_single_trip_for, cache_dir=tmp_path)
    result = numsim.Engine().run(module, args)

    expected = np.full(32, 3, dtype=np.int32)
    expected[:8] = 0
    expected[8:16] = 1
    np.testing.assert_array_equal(result.outputs["inside"], expected)
    np.testing.assert_array_equal(result.outputs["after"], np.full(32, 9, dtype=np.int32))
    assert "live_mask_loop" in module.rust_source
    assert "loop_offset_loop" not in module.rust_source
    assert "For loop offset overflow" not in module.rust_source
