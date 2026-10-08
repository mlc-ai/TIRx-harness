from __future__ import annotations

import math

import numpy as np
import pytest

from tirx_harness import numsim
from tests.numsim.corpus.kernels.deepgemm import (
    TF32_HC_CONFIGS as NUMSIM_CONFIGS,
    _float32_to_bfloat16_bits,
    _numpy_tf32_hc_reference,
    _round_float32_to_tf32,
    prepare_tf32_hc_case as prepare_numsim_case,
)
from tests.numsim.microtests.harness import NUMSIM_GPU_MARK, require_numsim_gpu
from tests.numsim.support._tirx_kernels import config_params, load_tirx_kernel
from tests.numsim.support.three_way import run_three_way_case
from tirx_harness.numsim.bindings import _decode_tensor_maps, _tensor_map_base_array
from tirx_harness.numsim.transpiler.frontend import analyze

_tf32_hc = load_tirx_kernel("deepgemm_sm100_tf32_hc_prenorm_gemm")
CONFIGS = _tf32_hc.CONFIGS
get_kernel = _tf32_hc.get_kernel


def test_tf32_hc_numsim_corpus_covers_unsplit_and_multi_split_paths():
    assert {config["num_splits"] == 1 for config in NUMSIM_CONFIGS} == {False, True}
    assert all(config["k"] % 64 == 0 for config in NUMSIM_CONFIGS)


@pytest.mark.parametrize("config", CONFIGS, ids=lambda config: config["label"])
def test_all_tf32_hc_configs_are_supported(config):
    kernel = get_kernel(**config)
    spec = analyze(kernel)

    assert spec.unsupported == ()


@pytest.mark.parametrize("config", NUMSIM_CONFIGS, ids=lambda config: config["label"])
def test_prepare_tf32_hc_numsim_case_is_deterministic_and_full_launch(config):
    case = prepare_numsim_case(**config_params(config))
    repeated = prepare_numsim_case(**config_params(config))
    topology = analyze(case.kernel).topology

    assert topology.clusters == config["num_splits"] * math.ceil(config["m"] / 64)
    assert topology.ctas_per_cluster == 1
    assert topology.warps_per_cta == 8
    np.testing.assert_array_equal(case.args["a"], repeated.args["a"])
    np.testing.assert_array_equal(case.args["b"], repeated.args["b"])
    b = case.args["b"].reshape(config["n"], config["k"])
    b_bits = b.view(np.uint32)
    assert np.any((b_bits & np.uint32(0x1FFF)) != 0)
    assert case.args["b"].shape == (config["n"] * config["k"],)
    b_map = case.args["b_map"]
    descriptors = _decode_tensor_maps(b_map)
    assert len(descriptors) == 1
    assert descriptors[0].dtype == "tf32"
    assert descriptors[0].address == case.args["b"].__array_interface__["data"][0]
    assert np.shares_memory(_tensor_map_base_array(b_map), case.args["b"])
    np.testing.assert_array_equal(case.reference()["D"], repeated.reference()["D"])
    np.testing.assert_array_equal(case.reference()["sqr_sum"], repeated.reference()["sqr_sum"])


def test_first_tf32_hc_case_matches_independent_reference_in_numsim():
    case = prepare_numsim_case(**config_params(NUMSIM_CONFIGS[0]))

    report = numsim.run_case(case)

    assert report.ok, report.mismatches[0].render()


def test_tf32_rounding_uses_round_to_nearest_even():
    one = np.float32(1.0)
    tf32_ulp = np.float32(2**-10)
    values = np.array(
        [one + tf32_ulp / 2, one + tf32_ulp + tf32_ulp / 2, -(one + tf32_ulp / 2)], dtype=np.float32
    )
    expected = np.array([one, one + 2 * tf32_ulp, -one], dtype=np.float32)
    np.testing.assert_array_equal(_round_float32_to_tf32(values), expected)


def test_tf32_hc_accepts_one_k_block_per_split():
    kernel = get_kernel(m=13, n=24, k=128, num_splits=2, seed=0)

    topology = analyze(kernel).topology
    assert topology.clusters == 2


def test_multi_split_reference_partials_reduce_to_unsplit_result():
    rng = np.random.default_rng(9)
    a = _float32_to_bfloat16_bits(
        rng.integers(-3, 4, size=(3, 256), dtype=np.int16).astype(np.float32) / 8
    )
    b = rng.standard_normal((5, 256), dtype=np.float32) * np.float32(0.25)
    unsplit_d, unsplit_sqr = _numpy_tf32_hc_reference(a, b, num_splits=1)
    partial_d, partial_sqr = _numpy_tf32_hc_reference(a, b, num_splits=3)

    np.testing.assert_allclose(partial_d.sum(axis=0), unsplit_d, rtol=2e-6, atol=2e-6)
    np.testing.assert_allclose(partial_sqr.sum(axis=0), unsplit_sqr, rtol=0, atol=0)


@NUMSIM_GPU_MARK
@pytest.mark.parametrize("config", NUMSIM_CONFIGS, ids=lambda config: config["label"])
def test_every_tf32_hc_numsim_config_matches_gpu_and_reference(pytestconfig, tmp_path, config):
    require_numsim_gpu(pytestconfig)

    report = run_three_way_case(prepare_numsim_case(**config_params(config)), cache_dir=tmp_path)

    report.require_ok()
