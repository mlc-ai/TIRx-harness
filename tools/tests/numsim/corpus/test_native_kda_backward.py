from __future__ import annotations

import numpy as np
import pytest

from tests.numsim.corpus.kernels.native_kda_backward import prepare_native_kda_backward_case
from tests.numsim.microtests.harness import NUMSIM_GPU_MARK, require_numsim_gpu
from tests.numsim.support.three_way import run_three_way_case
from tirx_harness.numsim import NumSimResult


@pytest.fixture(scope="module")
def backward_case():
    return prepare_native_kda_backward_case()


def test_backward_oracle_rejects_exp_of_log2_gate(backward_case):
    gate = backward_case.args["g"]
    expected = backward_case.reference()["dh0"]
    # g stores a log2 gate. For one token, dh0 is linear in the state decay;
    # replace exp2(g) with exp(g), without duplicating the recurrence oracle.
    wrong = expected.reshape(gate.size, -1) * (np.exp(gate) / np.exp2(gate))[:, None]
    with pytest.raises(AssertionError, match="NumSim comparison failed"):
        NumSimResult({"dh0": expected}).assert_close(
            {"dh0": wrong.reshape(expected.shape)},
            {"dh0": backward_case.comparisons["dh0"]},
        )


@NUMSIM_GPU_MARK
def test_backward_matches_gpu_and_reference(pytestconfig, backward_case):
    require_numsim_gpu(pytestconfig)
    report = run_three_way_case(backward_case, cache_dir=None)
    report.require_ok()
