from __future__ import annotations

import numpy as np
import pytest

from tests.numsim.corpus.kernels.attention import (
    FLASH_ATTENTION_BACKWARD_CONFIGS,
    prepare_flash_attention_backward_case,
)
from tirx_harness import numsim
from tirx_harness.numsim.transpiler.frontend import analyze


@pytest.mark.parametrize(
    "config", FLASH_ATTENTION_BACKWARD_CONFIGS, ids=lambda config: config["label"]
)
def test_flash_attention_backward_case_is_deterministic_and_full_launch(config):
    params = {name: value for name, value in config.items() if name != "label"}
    case = prepare_flash_attention_backward_case(**params)
    repeated = prepare_flash_attention_backward_case(**params)
    topology = analyze(case.kernel).topology

    assert topology.clusters >= 1
    assert topology.ctas_per_cluster == 2
    assert topology.warps_per_cta == 16
    for name in ("Q_g", "K_g", "V_g", "dO_g", "LSE_g", "dpsum_g"):
        np.testing.assert_array_equal(case.args[name], repeated.args[name])
    for name in case.outputs:
        np.testing.assert_array_equal(case.reference()[name], repeated.reference()[name])


@pytest.mark.parametrize(
    "config", FLASH_ATTENTION_BACKWARD_CONFIGS, ids=lambda config: config["label"]
)
def test_flash_attention_backward_matches_independent_numpy_backward(config, tmp_path_factory):
    params = {name: value for name, value in config.items() if name != "label"}
    case = prepare_flash_attention_backward_case(**params)
    module = numsim.transpile(
        case.kernel,
        cache_dir=tmp_path_factory.getbasetemp() / "flash-attention-backward-numsim",
    )
    result = numsim.Engine().run(module, case.args, outputs=case.outputs)

    numsim.compare(result, case.reference(), tolerances=case.comparisons).require_ok()
