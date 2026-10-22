from __future__ import annotations

import numpy as np
import pytest
import tirx_kernels.tirx_lite as txl
from tvm import tirx
from tvm.script import tirx as T

from tirx_harness.numsim import checkers
from tirx_harness.numsim.errors import NumSimExecutionError


def symbolic_state(state_in, state_out):
    # Match the recurrence's rank-only parameter annotations and permit aliasing.
    lane = txl.lane_id()
    value = txl.local_scalar("uint32")
    txl.ptx.ld.global_.b32(value, state_in.ptr_to([0, 0, 0, lane]))
    txl.ptx.st.global_.b32(state_out.ptr_to([0, 0, 0, lane]), value)


symbolic_state.__annotations__ = {
    "state_in": txl.gptr["float32", 4],
    "state_out": txl.gptr["float32", 4],
}
symbolic_state = txl.kernel(warps=1, grid=1)(symbolic_state)


_ROWS = tirx.Var("rows", "int64")


@T.prim_func
def shared_rows(source: T.Buffer((_ROWS, 32), "float32"), output: T.Buffer((_ROWS, 32), "float32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    output[0, lane] = source[0, lane]


@T.prim_func
def explicit_rows(rows: T.int32, output: T.Buffer((rows, 32), "int32")):
    T.device_entry()
    cta = T.cta_id([rows])
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    output[cta, lane] = lane


@T.prim_func
def symbolic_tensor_map(output: T.Buffer((_ROWS, 32), "int32")):
    tensor_map: T.let[T.TensorMap()] = T.tvm_stack_alloca("tensormap", 1)
    T.tensormap_encode_tiled(
        tensor_map,
        output.data,
        32,
        _ROWS,
        128,
        32,
        1,
        1,
        1,
        descriptor_dtype="int32",
        rank=2,
    )
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    output[0, lane] = lane


@T.prim_func
def symbolic_race(output: T.Buffer((T.int64(), 32), "int32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    output[0, 0] = lane


CHECKERS = [checkers._run_synccheck, checkers._run_racecheck]


@pytest.mark.parametrize("run", CHECKERS, ids=["synccheck", "racecheck"])
def test_input_shapes_are_bound_before_typed_tensor_map_normalization(run, tmp_path):
    run(
        symbolic_tensor_map,
        inputs={"output": np.zeros((3, 32), dtype=np.int32)},
        cache_dir=tmp_path,
    ).require_clean()


@pytest.mark.parametrize("run", CHECKERS, ids=["synccheck", "racecheck"])
def test_rank_only_input_shapes_are_bound_and_aliases_preserved(run, tmp_path):
    state = np.zeros((1, 48, 128, 128), dtype=np.float32)
    report = run(
        symbolic_state.func, inputs={"state_in": state, "state_out": state}, cache_dir=tmp_path
    )
    report.require_clean()
    native = report.to_dict()["native"]
    assert native["input"]["bindings"] == ["state_in", "state_out"]


@pytest.mark.parametrize("run", CHECKERS, ids=["synccheck", "racecheck"])
def test_explicit_shape_scalar_is_checked_and_recorded(run, tmp_path):
    inputs = {"rows": np.int32(2), "output": np.zeros((2, 32), dtype=np.int32)}
    report = run(explicit_rows, inputs=inputs, cache_dir=tmp_path)
    report.require_clean()
    native = report.to_dict()["native"]
    assert native["input"]["bindings"] == ["output", "rows"]
    assert native["input"]["scalars"]["rows"]["value"]["decimal"] == "2"
    with pytest.raises(NumSimExecutionError, match="disagrees with scalar input"):
        run(explicit_rows, inputs={**inputs, "rows": np.int32(3)}, cache_dir=tmp_path)


def test_bound_shapes_do_not_hide_a_race(tmp_path):
    report = checkers._run_racecheck(
        symbolic_race, inputs={"output": np.zeros((2, 32), dtype=np.int32)}, cache_dir=tmp_path
    )
    assert report.verdict == "error", report.format()


@pytest.mark.parametrize("run", CHECKERS, ids=["synccheck", "racecheck"])
def test_shared_shape_dimensions_must_agree(run, tmp_path):
    with pytest.raises(NumSimExecutionError, match="input shapes disagree"):
        run(
            shared_rows,
            inputs={
                "source": np.zeros((2, 32), dtype=np.float32),
                "output": np.zeros((3, 32), dtype=np.float32),
            },
            cache_dir=tmp_path,
        )


def test_shape_specialization_changes_artifact_but_data_does_not(tmp_path):
    def execute(rows, value):
        state = np.full((rows, 1, 1, 32), value, dtype=np.float32)
        report = checkers._run_racecheck(
            symbolic_state.func, inputs={"state_in": state, "state_out": state}, cache_dir=tmp_path
        )
        report.require_clean()
        return report.to_dict()["native"]

    first = execute(1, 0)
    changed_data = execute(1, 1)
    changed_shape = execute(2, 1)
    assert first["engine"]["artifact_key"] == changed_data["engine"]["artifact_key"]
    assert first["input"]["digest"] != changed_data["input"]["digest"]
    assert first["engine"]["artifact_key"] != changed_shape["engine"]["artifact_key"]
