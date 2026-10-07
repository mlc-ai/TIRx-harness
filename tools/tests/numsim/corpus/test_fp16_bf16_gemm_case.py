from __future__ import annotations

import numpy as np
import pytest

from tests.numsim.corpus.kernels.gemm import FP16_BF16_CONFIGS as NUMSIM_CONFIGS
from tests.numsim.corpus.kernels.gemm import _bfloat16_bits_to_float32, _float32_to_bfloat16_bits
from tests.numsim.corpus.kernels.gemm import prepare_fp16_bf16_case as prepare_numsim_case
from tests.numsim.microtests.harness import NUMSIM_GPU_MARK, require_numsim_gpu
from tests.numsim.support.three_way import run_three_way_case
from tirx_harness.numsim.transpiler.frontend import analyze


def test_gemm_numsim_corpus_matches_bootstrap_target():
    assert {config["dtype"] for config in NUMSIM_CONFIGS} == {"fp16", "bf16"}
    for config in NUMSIM_CONFIGS:
        assert (config["M"], config["N"], config["K"]) == (256, 2048, 64)


@pytest.mark.parametrize("config", NUMSIM_CONFIGS, ids=lambda config: config["label"])
def test_prepare_gemm_numsim_case_is_deterministic_and_full_launch(config):
    params = {key: value for key, value in config.items() if key != "label"}
    case = prepare_numsim_case(**params)
    repeated = prepare_numsim_case(**params)
    spec = analyze(case.kernel)

    assert spec.topology.clusters == 8
    assert spec.topology.ctas_per_cluster == 2
    assert spec.topology.warps_per_cta == 8
    assert case.subset is None
    assert case.args["a"].shape == (256, 64)
    assert case.args["b"].shape == (2048, 64)
    assert case.args["d"].shape == (256, 2048)
    np.testing.assert_array_equal(case.args["a"], repeated.args["a"])
    np.testing.assert_array_equal(case.args["b"], repeated.args["b"])
    np.testing.assert_array_equal(case.reference()["D"], repeated.reference()["D"])


def test_independent_bfloat16_codec_rounds_to_nearest_even():
    values = np.array([1.0, -2.5, 1.00390625, 1.01171875], dtype=np.float32)
    encoded = _float32_to_bfloat16_bits(values)
    decoded = _bfloat16_bits_to_float32(encoded)

    np.testing.assert_array_equal(decoded, np.array([1.0, -2.5, 1.0, 1.015625], dtype=np.float32))


@NUMSIM_GPU_MARK
@pytest.mark.parametrize("config", NUMSIM_CONFIGS, ids=lambda config: config["label"])
def test_every_gemm_numsim_config_matches_gpu_and_reference(pytestconfig, tmp_path, config):
    require_numsim_gpu(pytestconfig)
    params = {key: value for key, value in config.items() if key != "label"}

    report = run_three_way_case(prepare_numsim_case(**params), cache_dir=tmp_path)

    report.require_ok()
