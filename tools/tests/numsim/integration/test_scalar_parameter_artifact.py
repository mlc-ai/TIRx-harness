from __future__ import annotations

import numpy as np
import pytest

from tirx_harness import numsim
from tirx_harness.numsim.bindings import prepare_bindings
from tvm.script import tirx as T


@T.prim_func
def scalar_parameter_add(offset: T.int32, output: T.Buffer((32,), "int32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    output[lane] = offset + lane


@T.prim_func
def second_scalar_parameter_add(offset: T.int32, second: T.Buffer((32,), "int32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    second[lane] = offset - lane


@T.prim_func
def scalar_bound_shape(rows: T.int32, input_ptr: T.handle, output_ptr: T.handle):
    input_buffer = T.match_buffer(input_ptr, (rows, 32), "float32")
    output_buffer = T.match_buffer(output_ptr, (rows, 32), "float32")
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    output_buffer[0, lane] = input_buffer[0, lane] + T.float32(1)


def test_plain_and_numpy_int32_scalar_parameters_affect_generated_code(tmp_path):
    output = np.zeros(32, dtype=np.int32)
    module = numsim.transpile(scalar_parameter_add, cache_dir=tmp_path)

    plain = numsim.Engine().run(module, {"offset": 7, "output": output})
    np.testing.assert_array_equal(plain.outputs["output"], np.arange(32, dtype=np.int32) + 7)

    output.fill(0)
    explicit = numsim.Engine().run(module, {"offset": np.int32(-3), "output": output})
    np.testing.assert_array_equal(explicit.outputs["output"], np.arange(32, dtype=np.int32) - 3)
    assert module.spec.kernels[0].scalars[0].name == "offset"
    assert module.spec.kernels[0].scalars[0].dtype == "int32"
    assert "scalar_0: i32" in module.rust_source
    assert "buffers.scalar_0" in module.rust_source
    assert "fn extract_scalar_i32(" not in module.rust_source


def test_scalar_shape_consistency_is_checked_by_engine_api(tmp_path):
    source = np.arange(5 * 32, dtype=np.float32).reshape(5, 32)
    output = np.zeros_like(source)
    module = numsim.transpile(scalar_bound_shape, cache_dir=tmp_path)

    result = numsim.Engine().run(
        module,
        {"rows": 5, "input_buffer": source, "output_buffer": output},
        outputs=("output_buffer",),
    )
    np.testing.assert_array_equal(result.outputs["output_buffer"][0], source[0] + 1)
    assert "validate_shape_scalar(" in module.rust_source
    assert "cannot be used as a buffer extent" not in module.rust_source
    assert "disagrees with bound buffer extent" not in module.rust_source

    with pytest.raises(numsim.NumSimExecutionError, match="disagrees with bound buffer extent"):
        numsim.Engine().run(
            module,
            {"rows": 4, "input_buffer": source, "output_buffer": output},
        )


def test_generated_artifact_rejects_missing_scalar_parameter(tmp_path):
    output = np.zeros(32, dtype=np.int32)
    module = numsim.transpile(scalar_parameter_add, cache_dir=tmp_path)

    with pytest.raises(numsim.NumSimExecutionError, match="missing required bindings.*offset"):
        numsim.Engine().run(module, {"output": output})


def test_generated_artifact_rejects_wrong_scalar_dtype(tmp_path):
    output = np.zeros(32, dtype=np.int32)
    module = numsim.transpile(scalar_parameter_add, cache_dir=tmp_path)

    with pytest.raises(
        numsim.NumSimExecutionError, match="scalar 'offset' requires dtype int32, got uint32"
    ):
        numsim.Engine().run(module, {"offset": np.uint32(7), "output": output})


def test_scalar_parameter_range_is_checked_before_execution(tmp_path):
    output = np.zeros(32, dtype=np.int32)
    module = numsim.transpile(scalar_parameter_add, cache_dir=tmp_path)

    with pytest.raises(numsim.NumSimExecutionError, match="outside int32 range"):
        numsim.Engine().run(module, {"offset": 1 << 31, "output": output})

    payload = prepare_bindings({"offset": np.int32(0), "output": output}).to_payload()
    payload["scalars"]["offset"]["value"] = 1 << 31
    with pytest.raises(ValueError, match="outside int32 range"):
        module.load().run(payload, None)


def test_multi_kernel_scalar_names_are_phase_qualified(tmp_path):
    module = numsim.transpile(
        [scalar_parameter_add, second_scalar_parameter_add], cache_dir=tmp_path
    )
    result = numsim.Engine().run(
        module,
        {
            "k0:offset": 2,
            "k0:output": np.zeros(32, dtype=np.int32),
            "k1:offset": 5,
            "k1:second": np.zeros(32, dtype=np.int32),
        },
    )

    np.testing.assert_array_equal(result.outputs["k0:output"], np.arange(32, dtype=np.int32) + 2)
    np.testing.assert_array_equal(result.outputs["k1:second"], 5 - np.arange(32, dtype=np.int32))
