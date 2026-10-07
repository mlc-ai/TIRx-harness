from __future__ import annotations

import numpy as np
import pytest
from tvm.script import tirx as T

from tests.numsim.support.kernels import raw_tma_roundtrip
from tests.numsim.support.multi_kernels import (
    ambiguous_alias_first,
    ambiguous_alias_second,
    consume_intermediate,
    same_names_first,
    same_names_second,
    typed_pointer_first,
    typed_pointer_second,
    write_intermediate,
)
from tirx_harness import numsim
from tirx_harness.numsim.api import ExecutionSubset
from tirx_harness.numsim.transpiler.frontend import analyze
from tirx_harness.numsim.transpiler.artifact_template import emit_rust_module


@T.prim_func
def export_global_pointer_words(
    source: T.Buffer((32,), "uint32"), pointers: T.Buffer((32,), "uint64")
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    pointers[lane] = T.reinterpret("uint64", source.ptr_to([lane]))


@T.prim_func
def consume_global_pointer_words(
    pointers: T.Buffer((32,), "uint64"), output: T.Buffer((32,), "uint32")
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    T.ptx.ld.global_.u32(output[lane], T.reinterpret("handle", pointers[lane]))


def test_pointer_words_keep_allocation_binding_across_phases(tmp_path):
    source = np.arange(32, dtype=np.uint32) + 100
    pointers = np.zeros(32, dtype=np.uint64)
    module = numsim.transpile(
        [export_global_pointer_words, consume_global_pointer_words], cache_dir=tmp_path
    )
    result = numsim.Engine().run(
        module,
        {
            "k0:source": source,
            "k0:pointers": pointers,
            "k1:pointers": pointers,
            "k1:output": np.zeros(32, dtype=np.uint32),
        },
        outputs=("k1:output",),
    )
    np.testing.assert_array_equal(result.outputs["k1:output"], source)
    np.testing.assert_array_equal(pointers, source.ctypes.data + np.arange(32, dtype=np.uint64) * 4)


def test_two_launches_share_one_rust_owned_global_memory(tmp_path):
    intermediate = np.zeros(32, dtype=np.float32)
    output = np.zeros((2, 32), dtype=np.float32)

    module = numsim.transpile([write_intermediate, consume_intermediate], cache_dir=tmp_path)
    result = numsim.Engine().run(
        module,
        {"k0:intermediate": intermediate, "k1:intermediate": intermediate, "k1:output": output},
        outputs=("k1:output",),
    )

    expected_intermediate = np.arange(1, 33, dtype=np.float32)
    expected_output = np.stack((expected_intermediate * 2, expected_intermediate * 2 + 1))
    np.testing.assert_array_equal(intermediate, expected_intermediate)
    np.testing.assert_array_equal(result.outputs["k1:output"], expected_output)
    assert module.load().metadata()["kernel_count"] == 2
    assert module.load().metadata()["warp_counts"] == [1, 2]
    assert result.stats["task_count"] == 3
    assert result.stats["completed_task_count"] == 3
    assert [phase["task_count"] for phase in result.stats["kernels"]] == [1, 2]
    assert [phase["topology"]["clusters"] for phase in result.stats["kernels"]] == [1, 2]
    assert "async fn kernel_0_warp_main" in module.rust_source
    assert "async fn kernel_1_warp_main" in module.rust_source
    assert "program counter" not in module.rust_source

    intermediate.fill(0)
    output.fill(0)
    default_outputs = numsim.Engine().run(
        module,
        {"k0:intermediate": intermediate, "k1:intermediate": intermediate, "k1:output": output},
    )
    assert set(default_outputs.outputs) == {"k0:intermediate", "k1:intermediate", "k1:output"}
    np.testing.assert_array_equal(default_outputs.outputs["k0:intermediate"], expected_intermediate)
    np.testing.assert_array_equal(default_outputs.outputs["k1:output"], expected_output)


def test_multi_kernel_subset_is_selected_per_phase(tmp_path):
    intermediate = np.zeros(32, dtype=np.float32)
    output = np.full((2, 32), -1, dtype=np.float32)
    module = numsim.transpile([write_intermediate, consume_intermediate], cache_dir=tmp_path)

    result = numsim.Engine().run(
        module,
        {"k0:intermediate": intermediate, "k1:intermediate": intermediate, "k1:output": output},
        outputs=("k1:output",),
        subset={
            0: ExecutionSubset(cluster_ids=[0]),
            1: ExecutionSubset(cluster_ids=[1]),
        },
    )

    expected_intermediate = np.arange(1, 33, dtype=np.float32)
    np.testing.assert_array_equal(intermediate, expected_intermediate)
    np.testing.assert_array_equal(result.outputs["k1:output"][0], -np.ones(32, dtype=np.float32))
    np.testing.assert_array_equal(result.outputs["k1:output"][1], expected_intermediate * 2 + 1)
    assert [phase["task_count"] for phase in result.stats["kernels"]] == [1, 1]

    with pytest.raises(ValueError, match="not broadcast"):
        numsim.Engine().run(
            module,
            {"k0:intermediate": intermediate, "k1:intermediate": intermediate, "k1:output": output},
            subset=ExecutionSubset(cluster_ids=[0]),
        )


def test_multi_kernel_cache_identity_is_order_sensitive_and_collision_free(tmp_path):
    first = numsim.transpile((write_intermediate, consume_intermediate), cache_dir=tmp_path)
    cached = numsim.transpile([write_intermediate, consume_intermediate], cache_dir=tmp_path)
    reversed_module = numsim.transpile(
        [consume_intermediate, write_intermediate], cache_dir=tmp_path
    )

    assert first.cache_key == cached.cache_key
    assert first.library_path == cached.library_path
    assert first.load() is cached.load()
    assert reversed_module.cache_key != first.cache_key
    assert reversed_module.library_path != first.library_path
    assert reversed_module.load() is not first.load()
    assert "struct Kernel0Buffers" in first.rust_source
    assert "struct Kernel1Buffers" in first.rust_source


def test_multi_kernel_rejects_an_alias_with_multiple_canonical_targets(tmp_path):
    module = numsim.transpile([ambiguous_alias_first, ambiguous_alias_second], cache_dir=tmp_path)

    with pytest.raises(numsim.NumSimExecutionError, match="input alias 'shared' is ambiguous"):
        numsim.Engine().run(module, {"shared": np.zeros(32, dtype=np.float32)})


def test_multi_kernel_rejects_an_ambiguous_output_alias(tmp_path):
    first = np.zeros(32, dtype=np.float32)
    second = np.zeros(32, dtype=np.float32)
    module = numsim.transpile([ambiguous_alias_first, ambiguous_alias_second], cache_dir=tmp_path)

    with pytest.raises(numsim.NumSimExecutionError, match="output alias 'shared' is ambiguous"):
        numsim.Engine().run(
            module,
            {"k0:shared": first, "k1:shared": second},
            outputs=("shared",),
        )


def _same_name_inputs(*, shared_intermediate: bool):
    input0 = np.arange(32, dtype=np.float32)
    input1 = np.arange(32, dtype=np.float32) + np.float32(100)
    intermediate0 = np.zeros(32, dtype=np.float32)
    intermediate1 = intermediate0 if shared_intermediate else np.full(32, 7, dtype=np.float32)
    output0 = np.zeros(32, dtype=np.float32)
    output1 = np.zeros(32, dtype=np.float32)
    return {
        "k0:input": input0,
        "k0:intermediate": intermediate0,
        "k0:output": output0,
        "k0:scale": 3,
        "k1:input": input1,
        "k1:intermediate": intermediate1,
        "k1:output": output1,
        "k1:scale": np.float32(0.5),
    }


def test_same_local_names_use_distinct_phase_bindings(tmp_path):
    module = numsim.transpile([same_names_first, same_names_second], cache_dir=tmp_path)
    inputs = _same_name_inputs(shared_intermediate=False)

    result = numsim.Engine().run(module, inputs)

    input0 = inputs["k0:input"]
    input1 = inputs["k1:input"]
    np.testing.assert_array_equal(result.outputs["k0:intermediate"], input0 * 3)
    np.testing.assert_array_equal(result.outputs["k0:output"], input0 + 1)
    np.testing.assert_array_equal(result.outputs["k1:output"], input1 + np.float32(3.5))
    assert set(result.outputs) == {
        "k0:input",
        "k0:intermediate",
        "k0:output",
        "k1:input",
        "k1:intermediate",
        "k1:output",
    }


def test_cross_phase_sharing_requires_the_same_explicit_backing(tmp_path):
    module = numsim.transpile([same_names_first, same_names_second], cache_dir=tmp_path)
    inputs = _same_name_inputs(shared_intermediate=True)

    result = numsim.Engine().run(module, inputs)

    input0 = inputs["k0:input"]
    input1 = inputs["k1:input"]
    np.testing.assert_array_equal(result.outputs["k0:intermediate"], input0 * 3)
    np.testing.assert_array_equal(result.outputs["k1:output"], input1 + input0 * np.float32(1.5))


def test_multi_kernel_rejects_unqualified_scalar_collisions_and_unknown_inputs(tmp_path):
    module = numsim.transpile([same_names_first, same_names_second], cache_dir=tmp_path)
    inputs = _same_name_inputs(shared_intermediate=False)
    inputs["scale"] = inputs.pop("k0:scale")

    with pytest.raises(numsim.NumSimExecutionError, match="input alias 'scale' is ambiguous"):
        numsim.Engine().run(module, inputs)

    inputs = _same_name_inputs(shared_intermediate=False)
    inputs["scsale"] = 4
    with pytest.raises(numsim.NumSimExecutionError, match="binding 'scsale' is unknown"):
        numsim.Engine().run(module, inputs)


def test_multi_kernel_rejects_duplicate_qualified_aliases(tmp_path):
    module = numsim.transpile([write_intermediate, consume_intermediate], cache_dir=tmp_path)
    intermediate = np.zeros(32, dtype=np.float32)
    output = np.zeros((2, 32), dtype=np.float32)
    inputs = {
        "k0:intermediate": intermediate,
        "k1:intermediate": intermediate,
        "k1:output": output,
        "output": output,
    }

    with pytest.raises(numsim.NumSimExecutionError, match="provided through multiple aliases"):
        numsim.Engine().run(module, inputs)


def test_single_kernel_keeps_unqualified_binding_names(tmp_path):
    module = numsim.transpile(same_names_first, cache_dir=tmp_path)
    source = np.arange(32, dtype=np.float32)
    intermediate = np.zeros(32, dtype=np.float32)
    output = np.zeros(32, dtype=np.float32)

    result = numsim.Engine().run(
        module,
        {
            "input": source,
            "intermediate": intermediate,
            "output": output,
            "scale": 2,
        },
    )

    assert set(result.outputs) == {"input", "intermediate", "output"}
    np.testing.assert_array_equal(result.outputs["intermediate"], source * 2)
    np.testing.assert_array_equal(result.outputs["output"], source + 1)


def test_multi_kernel_pointer_parameters_are_phase_qualified(tmp_path):
    module = numsim.transpile([typed_pointer_first, typed_pointer_second], cache_dir=tmp_path)
    source0 = np.arange(32, dtype=np.uint32)
    source1 = np.arange(32, dtype=np.uint32) + np.uint32(100)

    result = numsim.Engine().run(
        module,
        {
            "k0:pointer": source0,
            "k0:output": np.zeros(32, dtype=np.uint32),
            "k1:pointer": source1,
            "k1:output": np.zeros(32, dtype=np.uint32),
        },
    )

    def check() -> None:
        np.testing.assert_array_equal(result.outputs["k0:output"], source0)
        np.testing.assert_array_equal(result.outputs["k1:output"], source1 + np.uint32(1))

    check()
    assert '"k0:pointer"' in module.rust_source
    assert '"k1:pointer"' in module.rust_source


def test_multi_kernel_tensor_maps_use_phase_qualified_rust_extraction_names():
    funcs = [raw_tma_roundtrip, raw_tma_roundtrip]
    spec = analyze(funcs)

    source = emit_rust_module(spec, funcs)

    for kernel_index in range(2):
        assert f'"k{kernel_index}:input_map"' in source
        assert f'"k{kernel_index}:output_map"' in source
