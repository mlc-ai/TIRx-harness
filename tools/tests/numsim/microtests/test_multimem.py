from __future__ import annotations

import numpy as np
import pytest

from tests.numsim.microtests import multigpu
from tests.numsim.microtests.cases.multimem import MULTIMEM_CASES
from tests.numsim.microtests.harness import NUMSIM_GPU_MARK, require_numsim_gpu
from tests.numsim.microtests.multigpu import (
    mismatches,
    run_gpu_case,
    run_gpu_kernel,
    run_numsim_case,
)
from tests.numsim.support import multimem_kernels, multimem_litmus
from tirx_harness import numsim


def _require_multicast(pytestconfig: pytest.Config, world: int) -> None:
    require_numsim_gpu(pytestconfig)
    torch = pytest.importorskip("torch")
    if torch.cuda.device_count() < world:
        pytest.skip(f"multimem microtests need {world} GPUs")
    from cuda.bindings import driver

    (status,) = driver.cuInit(0)
    assert status == driver.CUresult.CUDA_SUCCESS
    status, supported = driver.cuDeviceGetAttribute(
        driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED, 0
    )
    if status != driver.CUresult.CUDA_SUCCESS or not supported:
        pytest.skip("multimem microtests need NVLink multicast (NVLS)")


@NUMSIM_GPU_MARK
@pytest.mark.parametrize("world", [2, 4])
@pytest.mark.parametrize("case", MULTIMEM_CASES, ids=lambda case: case.name)
def test_multimem_matches_gpu(case, world, pytestconfig, tmp_path):
    _require_multicast(pytestconfig, world)
    gpu = run_gpu_case(case.name, world)
    sim = run_numsim_case(case, world, cache_dir=tmp_path)
    report = mismatches(case, gpu, sim)
    assert not report, "\n".join(report)


_KERNELS = "tests.numsim.support.multimem_kernels"
_GEMM = (f"{_KERNELS}:gemm_all_reduce_two_shot", f"{_KERNELS}:gemm_inputs")


def _gemm(world: int, protocol: str, dtype: str, integer: bool):
    shape = multimem_kernels.GEMM_SHAPE
    kernel = (_GEMM[0], (*shape, world, dtype))
    inputs = (_GEMM[1], (*shape, world, protocol, dtype), {"integer": integer, "paired": True})
    gpu = run_gpu_kernel(kernel, inputs, world, ("c", "flag"))
    func = multimem_kernels.gemm_all_reduce_two_shot(*shape, world, dtype)
    bindings = multimem_kernels.gemm_inputs(*shape, world, protocol, dtype, integer=integer)
    return gpu, numsim.Engine().run(numsim.transpile(func), bindings, outputs=["c", "flag"])


def _rank_output(sim, name: str, rank: int) -> np.ndarray:
    return np.asarray(sim.outputs[numsim.rank_binding_name(name, rank)])


_LITMUS = "tests.numsim.support.multimem_litmus"


@NUMSIM_GPU_MARK
@pytest.mark.parametrize("world", [2, 4])
@pytest.mark.parametrize("kernel,inputs,modes,outputs", [
    pytest.param("message_passing", "message_passing_inputs", {}, ("out", "data", "flag"),
                 id="mp-release_acquire_sys"),
    pytest.param("message_passing", "message_passing_inputs",
                 {"arrive": "fence_sys_relaxed", "wait": "relaxed_fence_sys",
                  "alias": "after_wait"},
                 ("out", "data", "flag"), id="mp-fenced_relaxed"),
    pytest.param("message_passing", "message_passing_inputs", {"wait": "wait_until_sys"},
                 ("out", "data", "flag"), id="mp-wait_until_sys"),
    pytest.param("broadcast", "broadcast_inputs", {"read_first": 0, "alias": 1},
                 ("out", "data", "flag"), id="broadcast-raw"),
    pytest.param("broadcast", "broadcast_inputs", {"read_first": 1, "alias": 1},
                 ("out", "data", "flag"), id="broadcast-war"),
    pytest.param("concurrent_writes", "concurrent_write_inputs", {"op": "red_relaxed_sys"},
                 ("data",), id="writes-red_relaxed_sys"),
])
def test_ordering_litmus_matches_gpu(kernel, inputs, modes, outputs, world, pytestconfig,
                                     tmp_path):
    """The race-free litmus programs Racecheck proves clean compute on NVLS
    hardware exactly what NumSim computes."""
    _require_multicast(pytestconfig, world)
    gpu = run_gpu_kernel((f"{_LITMUS}:{kernel}", None),
                         (f"{_LITMUS}:{inputs}", (world,), modes), world, outputs)

    sim = numsim.Engine().run(
        numsim.transpile(getattr(multimem_litmus, kernel), cache_dir=tmp_path),
        getattr(multimem_litmus, inputs)(world, **modes), outputs=list(outputs),
    )
    for rank in range(world):
        for output in outputs:
            np.testing.assert_array_equal(
                _rank_output(sim, output, rank), gpu[rank][output],
                err_msg=f"{output}@rank{rank}",
            )


@NUMSIM_GPU_MARK
@pytest.mark.parametrize("world", [2, 4])
def test_ported_two_shot_all_reduce_matches_gpu(world, pytestconfig, tmp_path):
    _require_multicast(pytestconfig, world)
    kernel = (f"{_KERNELS}:two_shot_all_reduce", (world, world))
    inputs = (f"{_KERNELS}:two_shot_inputs", (world, world), {})
    gpu = run_gpu_kernel(kernel, inputs, world, ("data_out", "flag"))

    func = multimem_kernels.two_shot_all_reduce(world, world)
    sim = numsim.Engine().run(numsim.transpile(func, cache_dir=tmp_path),
                              multimem_kernels.two_shot_inputs(world, world),
                              outputs=["data_out", "flag"])
    for rank in range(world):
        for output in ("data_out", "flag"):
            np.testing.assert_array_equal(
                _rank_output(sim, output, rank).view(np.uint32),
                gpu[rank][output].view(np.uint32), err_msg=f"{output}@rank{rank}",
            )


@NUMSIM_GPU_MARK
@pytest.mark.parametrize("world", [2, 4])
@pytest.mark.parametrize("dtype", ["float32", "bfloat16", "float16"])
@pytest.mark.parametrize("protocol", ["flashinfer", "cutlass", "sys_fenced"])
def test_ported_gemm_all_reduce_matches_gpu_bitwise(protocol, dtype, world, pytestconfig):
    """Integer operands make every product and partial sum exact, so the
    tensor core's reduction order cannot show; C is then bitwise the GPU's."""
    _require_multicast(pytestconfig, world)
    gpu, sim = _gemm(world, protocol, dtype, integer=True)

    for rank in range(world):
        for output in ("c", "flag"):
            np.testing.assert_array_equal(
                _rank_output(sim, output, rank), gpu[rank][output],
                err_msg=f"{output}@rank{rank}",
            )


@NUMSIM_GPU_MARK
@pytest.mark.parametrize("world", [2, 4])
@pytest.mark.parametrize("dtype,tolerance", [
    ("float32", 1e-6), ("bfloat16", 1e-2), ("float16", 2e-3),
])
def test_ported_gemm_all_reduce_random_operands_agree_with_gpu(dtype, tolerance, world,
                                                                 pytestconfig):
    """NumSim reduces each tcgen05 dot product as one increasing-K binary32 FMA
    chain; the tensor core associates differently, so on random operands both
    are held to C's rounding error against the float64 sum instead of to each
    other's bits."""
    _require_multicast(pytestconfig, world)
    gpu, sim = _gemm(world, "sys_fenced", dtype, integer=False)
    shape = multimem_kernels.GEMM_SHAPE
    expected = multimem_kernels.gemm_reference(
        multimem_kernels.gemm_inputs(*shape, world, "sys_fenced", dtype, paired=True))
    bound = tolerance * np.abs(expected).max()

    for rank in range(world):
        for c in (gpu[rank]["c"], _rank_output(sim, "c", rank)):
            np.testing.assert_allclose(multimem_kernels.c_values(c, dtype), expected,
                                       rtol=0, atol=bound, err_msg=f"c@rank{rank}")
