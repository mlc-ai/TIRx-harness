"""sin/cos .2pi on the GPU and in NumSim, against PTX's cycles error bounds."""

import math

import numpy as np
import pytest

import tvm_tirx
from tests.numsim.microtests.harness import require_numsim_gpu, run_gpu_primfunc
from tirx_harness import numsim
from tvm_tirx.script import tirx as T

_COUNT = 256


def _trig_cycles_kernel():
    source = f"""
@T.function
def trig_cycles(source: T.Tensor(({_COUNT},), "float32"), result: T.Tensor((2, {_COUNT}), "float32")):
    T.device_entry(launch=T.cuda.LaunchConfig(grid={_COUNT // 32}, block=32))
    index = T.cuda.block_idx("x") * 32 + T.cuda.thread_idx("x")
    T.ptx["sin.approx.2pi.ftz.f32"](result[0, index], source[index])
    T.ptx["cos.approx.2pi.ftz.f32"](result[1, index], source[index])
"""
    return tvm_tirx.script.from_source(source, extra_vars={"T": T})


def _turns() -> np.ndarray:
    exact = [0.0, -0.0, 0.25, -0.25, 0.5, -0.5, 0.75, 1.0, -1.0, 0.125, 2.5, -49.75, 50.0]
    rng = np.random.default_rng(0)
    near = rng.uniform(-1.0, 1.0, 128 - len(exact))
    far = rng.uniform(-50.0, 50.0, _COUNT - 128)
    return np.concatenate([exact, near, far]).astype(np.float32)


@pytest.mark.numsim_gpu
def test_trig_cycles_match_the_ptx_bound_on_gpu_and_numsim(pytestconfig, tmp_path):
    # PTX ISA 9.5 needs NVRTC 13.5.
    require_numsim_gpu(pytestconfig, min_nvrtc=(13, 5))
    kernel = _trig_cycles_kernel()
    turns = _turns()
    inputs = {"source": turns, "result": np.zeros((2, _COUNT), np.float32)}
    device = run_gpu_primfunc(kernel, inputs, outputs=("result",))["result"]
    simulated = (
        numsim.Engine()
        .run(numsim.transpile(kernel, cache_dir=tmp_path), inputs, outputs=("result",))
        .outputs["result"]
    )

    angle = 2.0 * math.pi * np.fmod(turns.astype(np.float64), 1.0)
    bound = np.where(np.abs(turns) <= 1.0, 2**-20, 2**-14.2)
    for row, exact in enumerate((np.sin(angle), np.cos(angle))):
        assert np.all(np.abs(device[row] - exact) <= bound), ("gpu", row)
        assert np.all(np.abs(simulated[row] - exact) <= bound), ("numsim", row)
    # Signed zero inputs keep their sign through sin.
    np.testing.assert_array_equal(np.signbit(device[0, :2]), [False, True])
    np.testing.assert_array_equal(np.signbit(simulated[0, :2]), [False, True])
