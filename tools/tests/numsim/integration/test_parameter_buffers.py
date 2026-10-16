import numpy as np
from dataclasses import replace
import pytest
import tvm
from tvm.script import tirx as T

from tirx_harness import numsim
from tirx_harness.numsim.errors import NumSimExecutionError, UnsupportedTIRxError
from tirx_harness.numsim.transpiler.frontend import attach_source_nodes, module_spec_from_manifest
from tirx_harness.numsim.checkers import _run_racecheck, _run_synccheck


def parameter_kernel(dtype="int64", extent=72, statement="output[thread] = parameters[index]"):
    return tvm.script.from_source(
        f'''
@T.prim_func
def main(output: T.Buffer((32,), "{dtype}"),
         parameters: T.Buffer(({extent},), "{dtype}", scope="local.param"), index: T.int32):
    T.device_entry()
    thread = T.thread_id([32])
    {statement}
''',
        {"T": T},
    )


@pytest.mark.parametrize("dtype", ["int32", "int64", "uint32", "uint64", "float32", "float64"])
@pytest.mark.parametrize("extent", [4, 72])
def test_by_value_parameters_match_independent_array_reads(dtype, extent, tmp_path):
    values = np.arange(extent, dtype=dtype) * 7
    artifact = numsim.transpile(parameter_kernel(dtype, extent), cache_dir=tmp_path)
    inputs = {
        "output": np.zeros(32, dtype=dtype),
        "parameters": values,
        "index": np.int32(extent - 1),
    }
    result = numsim.Engine().run(artifact, inputs, outputs=["output"])
    np.testing.assert_array_equal(result.outputs["output"], np.full(32, values[-1], dtype=dtype))


def test_by_value_parameter_replicas_remain_rank_private(tmp_path):
    artifact = numsim.transpile(parameter_kernel(), cache_dir=tmp_path)
    rows = [
        {
            "output": np.zeros(32, np.int64),
            "parameters": np.arange(72, dtype=np.int64) + rank * 100,
            "index": np.int32(rank + 3),
        }
        for rank in range(4)
    ]
    result = numsim.Engine().run(artifact, rows, outputs=["output"])
    for rank, row in enumerate(rows):
        np.testing.assert_array_equal(
            result.outputs[numsim.rank_binding_name("output", rank)],
            np.full(32, row["parameters"][rank + 3], np.int64),
        )


@pytest.mark.parametrize(
    "statement",
    [
        "parameters[0] = T.int64(7)",
        "T.ptx.mov.b64(parameters[0], T.int64(7))",
    ],
)
def test_writing_a_by_value_parameter_is_rejected(statement, tmp_path):
    with pytest.raises(UnsupportedTIRxError, match="read-only"):
        numsim.transpile(parameter_kernel(statement=statement), cache_dir=tmp_path)


def test_parameter_addresses_cannot_escape(tmp_path):
    function = parameter_kernel(
        statement='output[thread] = T.reinterpret("int64", parameters.ptr_to([0]))'
    )
    with pytest.raises(UnsupportedTIRxError, match="addresses cannot escape"):
        numsim.transpile(function, cache_dir=tmp_path)


def test_ptx_can_read_a_parameter_value_without_writing_it(tmp_path):
    function = parameter_kernel(
        statement='value = T.local_scalar("int64")\n    T.ptx.mov.b64(value, parameters[index])\n    output[thread] = value'
    )
    artifact = numsim.transpile(function, cache_dir=tmp_path)
    values = np.arange(72, dtype=np.int64) - 50
    result = numsim.Engine().run(
        artifact,
        {"output": np.zeros(32, np.int64), "parameters": values, "index": np.int32(7)},
        outputs=["output"],
    )
    np.testing.assert_array_equal(result.outputs["output"], np.full(32, values[7], np.int64))


def test_parameter_snapshot_does_not_alias_other_host_bindings(tmp_path):
    function = parameter_kernel(
        statement="if thread == 0:\n        output[0] = T.int64(999)\n        output[1] = parameters[0]"
    )
    artifact = numsim.transpile(function, cache_dir=tmp_path)
    backing = np.arange(72, dtype=np.int64) + 13
    original = backing.copy()
    result = numsim.Engine().run(
        artifact,
        {"output": backing[:32], "parameters": backing, "index": np.int32(0)},
        outputs=["output", "parameters"],
    )
    assert result.outputs["output"][0] == 999
    assert result.outputs["output"][1] == original[0]
    np.testing.assert_array_equal(result.outputs["parameters"], original)


@pytest.mark.parametrize("checker", [_run_racecheck, _run_synccheck])
def test_native_checkers_use_the_same_parameter_snapshot(checker, tmp_path):
    function = parameter_kernel(
        statement="if thread == 0:\n        output[0] = T.int64(999)\n        output[1] = parameters[0]"
    )
    backing = np.arange(72, dtype=np.int64) + 13
    original = backing.copy()
    report = checker(
        function,
        inputs={"output": backing[:32], "parameters": backing, "index": np.int32(0)},
        cache_dir=tmp_path,
    )
    assert report.verdict == "clean", report.format()
    assert report.native_payload["findings"] == []
    assert report.native_payload["incomplete"] == []
    np.testing.assert_array_equal(backing, original)


@pytest.mark.parametrize("checker", [_run_racecheck, _run_synccheck])
def test_native_checkers_snapshot_aliased_parameters_on_every_rank(checker, tmp_path):
    function = parameter_kernel(
        statement="if thread == 0:\n        output[0] = T.int64(999)\n        output[1] = parameters[0]"
    )
    replicas = [np.arange(72, dtype=np.int64) + rank * 100 for rank in range(4)]
    originals = [replica.copy() for replica in replicas]
    report = checker(
        function,
        inputs=[
            {"output": replica[:32], "parameters": replica, "index": np.int32(0)}
            for replica in replicas
        ],
        cache_dir=tmp_path,
    )
    assert report.verdict == "clean", report.format()
    assert report.native_payload["findings"] == []
    assert report.native_payload["incomplete"] == []
    for replica, original in zip(replicas, originals):
        np.testing.assert_array_equal(replica, original)


def test_parameter_snapshot_metadata_survives_artifact_cache_reload(tmp_path):
    function = parameter_kernel()
    first = numsim.transpile(function, cache_dir=tmp_path)
    second = numsim.transpile(function, cache_dir=tmp_path)
    assert first.spec.parameter_buffers == (("parameters",),)
    assert second.spec.parameter_buffers == first.spec.parameter_buffers
    restored = module_spec_from_manifest(second.spec.to_manifest())
    assert restored.parameter_buffers == first.spec.parameter_buffers
    assert restored.host_abi["kernel_count"] == 1
    result = numsim.Engine().run(
        second,
        {
            "output": np.zeros(32, np.int64),
            "parameters": np.arange(72, dtype=np.int64),
            "index": np.int32(71),
        },
        outputs=["output"],
    )
    np.testing.assert_array_equal(result.outputs["output"], 71)


def test_source_attachment_rejects_missing_parameter_snapshot_metadata(tmp_path):
    function = parameter_kernel()
    module = numsim.transpile(function, cache_dir=tmp_path)
    stale = replace(module.spec, parameter_buffers=())
    with pytest.raises(ValueError, match="parameter snapshot metadata"):
        attach_source_nodes(stale, function)


def test_parameter_snapshot_is_distinct_from_an_ordinary_aliased_buffer(tmp_path):
    function = parameter_kernel()
    ordinary = tvm.script.from_source(
        function.script().replace('scope="local.param"', 'scope="global"'), {"T": T}
    )
    marked = numsim.transpile(function, cache_dir=tmp_path)
    unmarked = numsim.transpile(ordinary, cache_dir=tmp_path)
    assert marked.artifact.key != unmarked.artifact.key


def test_parameter_buffer_requires_the_declared_extent(tmp_path):
    artifact = numsim.transpile(parameter_kernel(), cache_dir=tmp_path)
    with pytest.raises(NumSimExecutionError, match="requires shape"):
        numsim.Engine().run(
            artifact,
            {
                "output": np.zeros(32, np.int64),
                "parameters": np.zeros(73, np.int64),
                "index": np.int32(0),
            },
        )


def test_parameter_snapshot_does_not_hide_noncontiguous_bindings(tmp_path):
    artifact = numsim.transpile(parameter_kernel(), cache_dir=tmp_path)
    with pytest.raises(NumSimExecutionError, match="requires contiguous storage"):
        numsim.Engine().run(
            artifact,
            {
                "output": np.zeros(32, np.int64),
                "parameters": np.arange(144, dtype=np.int64)[::2],
                "index": np.int32(0),
            },
        )


@pytest.mark.parametrize("index", [-1, 72])
def test_parameter_buffer_reads_are_bounds_checked(index, tmp_path):
    artifact = numsim.transpile(parameter_kernel(), cache_dir=tmp_path)
    with pytest.raises(NumSimExecutionError, match="bounds|outside"):
        numsim.Engine().run(
            artifact,
            {
                "output": np.zeros(32, np.int64),
                "parameters": np.arange(72, dtype=np.int64),
                "index": np.int32(index),
            },
        )
