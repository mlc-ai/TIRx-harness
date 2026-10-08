import numpy as np
import pytest

from tests.numsim.microtests.harness import (
    _compile_gpu,
    require_numsim_gpu,
    run_paired_primfunc,
)
from tests.numsim.runtime.test_atomic_f32_noftz import (
    ATOMIC_CASES,
    atomic_inputs,
    atomic_kernel,
    bulk_kernel,
)


@pytest.fixture(scope="module")
def noftz_compiler(pytestconfig):
    require_numsim_gpu(pytestconfig)
    try:
        _compile_gpu(atomic_kernel("atom", 1, "global"), arch="sm_100a")
    except RuntimeError as error:
        if "Illegal modifier '.noftz' for instruction 'atom'" not in str(error):
            raise
        pytest.skip("active CUDA compiler does not support PTX 9.4 atom.add.noftz.f32")


def _finite_inputs():
    inputs = atomic_inputs()
    # NaN result payloads are unspecified; CPU tests check their classification
    # separately. Keep the bitwise GPU differential on finite operands.
    for value in inputs.values():
        value[~np.isfinite(value)] = 0
    return inputs


@pytest.mark.numsim_gpu
def test_atomic_f32_noftz_matches_gpu(noftz_compiler, tmp_path):
    for kind, width, space in ATOMIC_CASES:
        run_paired_primfunc(
            atomic_kernel(kind, width, space),
            _finite_inputs(),
            outputs=("destination", "returned"),
            cache_dir=tmp_path,
        )


@pytest.mark.numsim_gpu
def test_bulk_f32_noftz_matches_gpu(noftz_compiler, tmp_path):
    inputs = _finite_inputs()
    run_paired_primfunc(
        bulk_kernel(),
        {
            "source": inputs["value"][:16],
            "destination": inputs["destination"][:16],
            "observed": np.zeros(1, np.float32),
        },
        outputs=("destination", "observed"),
        cache_dir=tmp_path,
    )
