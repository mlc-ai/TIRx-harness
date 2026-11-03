"""Stored global pointer bytes use normal local indexing."""

import numpy as np
import pytest

import tvm_tirx
from tests.numsim.support.execution import run_checked
from tirx_harness import numsim, racecheck, synccheck
from tirx_harness.numsim.errors import NumSimExecutionError
from tvm_tirx.script import tirx as T


def partial_array_case(mode):
    index = {"valid": "lane % 2", "uninitialized": "1 - lane % 2", "oob": "2"}[mode]
    return tvm_tirx.script.from_source(
        f"""
@T.function
def kernel(source: T.Tensor((32,), "uint32"), output: T.Tensor((32,), "uint32")):
    T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=32))
    lane = T.cuda.lane_id()
    pointers = T.alloc_local((2,), "uint64")
    pointers[lane % 2] = T.reinterpret(source.ptr_to([lane]), ty="uint64")
    T.cuda.cta_sync()
    T.ptx.ld.global_.u32(output[lane], T.reinterpret(pointers[{index}], ty="handle"))
    T.ptx.ld.global_.u32(output[lane], T.reinterpret(pointers[99], ty="handle"), pred=False, preserve_dst=True)
""",
        {"T": T},
    )


@pytest.mark.parametrize("mode", ["valid", "uninitialized", "oob"])
def test_pointer_array_initialization_and_bounds(mode, tmp_path):
    kernel = partial_array_case(mode)
    inputs = {"source": np.arange(32, dtype=np.uint32) + 100, "output": np.zeros(32, np.uint32)}
    message = "out-of-bounds" if mode == "uninitialized" else "outside|exceeds"
    for checker in (synccheck, racecheck):
        report = checker(kernel, {k: v.copy() for k, v in inputs.items()})
        if mode == "valid":
            report.require_clean()
        else:
            assert report.verdict == "error", report.format()
            if mode == "uninitialized":
                # The integer model materializes an uninitialized word as zero;
                # the warning and the later invalid-address error both remain.
                assert any(f.kind == "uninitialized_read" for f in report.findings), report.format()
                assert any(f.kind == "oob" for f in report.findings), report.format()
    module = numsim.transpile(kernel, cache_dir=tmp_path)
    if mode == "valid":
        result = numsim.Engine().run(module, inputs)
        np.testing.assert_array_equal(result.outputs["output"], inputs["source"])
    else:
        with pytest.raises(NumSimExecutionError, match=message):
            numsim.Engine().run(module, inputs)


@T.function
def pointer_array_alias(source: T.Tensor((64,), "uint32"), output: T.Tensor((32, 2), "uint32")):
    T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=32))
    lane = T.cuda.lane_id()
    pointers = T.alloc_local((2, 2), "uint64")
    alias = T.decl_tensor(
        T.ptr_byte_offset(pointers.data, 16, ty=T.Ptr("uint64", "local")),
        ty_args=[T.Tensor((2,), "uint64", scope="local")],
    )
    for index in T.serial(4):
        pointers[index // 2, index % 2] = T.reinterpret(source.ptr_to([lane + index]), ty="uint64")
    T.ptx.ld.global_.u32(output[lane, 0], T.reinterpret(alias[lane % 2], ty="handle"))
    alias[lane % 2] = T.reinterpret(source.ptr_to([lane + 32]), ty="uint64")
    T.cuda.cta_sync()
    T.ptx.ld.global_.u32(output[lane, 1], T.reinterpret(pointers[1, lane % 2], ty="handle"))


def test_pointer_array_alias_and_suspension(tmp_path):
    source = np.arange(64, dtype=np.uint32) + 100
    inputs = {"source": source, "output": np.zeros((32, 2), np.uint32)}
    result = run_checked(pointer_array_alias, inputs, cache_dir=tmp_path)
    expected = np.stack([source[np.arange(32) + 2 + np.arange(32) % 2], source[32:]], axis=1)
    np.testing.assert_array_equal(result.outputs["output"], expected)
