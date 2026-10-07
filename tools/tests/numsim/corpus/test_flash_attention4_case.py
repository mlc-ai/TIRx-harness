from __future__ import annotations

import math

import numpy as np
import pytest

from tests.numsim.corpus.kernels.attention import (
    FLASH_ATTENTION4_CONFIGS as NUMSIM_CONFIGS,
)
from tests.numsim.corpus.kernels.attention import (
    numpy_attention_reference,
    prepare_flash_attention4_case as prepare_numsim_case,
)
from tests.numsim.support._tirx_kernels import load_tirx_kernel
from tirx_harness.numsim.bindings import _decode_tensor_maps, _tensor_map_base_array
from tirx_harness.numsim.transpiler.frontend import analyze

get_kernel = load_tirx_kernel("flash_attention4").get_kernel


def _scalar_attention(q, k, v, *, causal):
    batch, q_len, q_heads, dim = q.shape
    kv_len = k.shape[1]
    kv_heads = k.shape[2]
    ratio = q_heads // kv_heads
    output = np.empty_like(q, dtype=np.float32)
    for b in range(batch):
        for qi in range(q_len):
            for h in range(q_heads):
                kh = h // ratio
                logits = []
                for ki in range(kv_len):
                    masked = causal and ki > qi + kv_len - q_len
                    dot = sum(float(q[b, qi, h, d]) * float(k[b, ki, kh, d]) for d in range(dim))
                    logits.append(-math.inf if masked else dot / math.sqrt(dim))
                maximum = max(logits)
                weights = [math.exp(value - maximum) for value in logits]
                denominator = sum(weights)
                for d in range(dim):
                    output[b, qi, h, d] = (
                        sum(weights[ki] * float(v[b, ki, kh, d]) for ki in range(kv_len))
                        / denominator
                    )
    return output.astype(np.float16)


def test_flash_attention_numsim_corpus_covers_every_gqa_ratio_and_causal_branch():
    assert len(NUMSIM_CONFIGS) == 12
    signatures = {
        (
            config["seq_len"],
            config["num_qo_heads"] // config["num_kv_heads"],
            config["is_causal"],
        )
        for config in NUMSIM_CONFIGS
    }
    assert signatures == {
        (256, ratio, causal) for ratio in (1, 2, 4, 8) for causal in (False, True)
    } | {(384, ratio, causal) for ratio in (1, 8) for causal in (False, True)}


@pytest.mark.parametrize("label", ["s256_mha_noncausal", "s256_gqa8_causal"])
def test_prepare_flash_attention_numsim_case_is_deterministic(label):
    config = next(item for item in NUMSIM_CONFIGS if item["label"] == label)
    params = {key: value for key, value in config.items() if key != "label"}
    case = prepare_numsim_case(**params)
    repeated = prepare_numsim_case(**params)

    q_descriptor = _decode_tensor_maps(case.args["Q_tensor_map"])[0]
    k_descriptor = _decode_tensor_maps(case.args["K_tensor_map"])[0]
    o_descriptor = _decode_tensor_maps(case.args["O_tensor_map"])[0]
    assert q_descriptor.global_shape == o_descriptor.global_shape
    assert q_descriptor.dtype == o_descriptor.dtype == "float16"
    assert math.prod(k_descriptor.global_shape) == 256 * params["num_kv_heads"] * 128
    assert analyze(case.kernel).topology.warps_per_cta == 16
    np.testing.assert_array_equal(
        _tensor_map_base_array(case.args["Q_tensor_map"]),
        _tensor_map_base_array(repeated.args["Q_tensor_map"]),
    )
    np.testing.assert_array_equal(case.reference()["O"], repeated.reference()["O"])


def test_flash_attention_corpus_has_expected_full_launch_topologies():
    expected_clusters = {
        "s256_mha_noncausal": 32,
        "s256_gqa8_noncausal": 32,
        "s384_mha_noncausal": 64,
        "s384_gqa8_noncausal": 48,
    }
    for label, clusters in expected_clusters.items():
        config = next(item for item in NUMSIM_CONFIGS if item["label"] == label)
        params = {key: value for key, value in config.items() if key not in {"label", "seed"}}
        topology = analyze(get_kernel(**params)).topology
        assert topology.clusters == clusters
        assert topology.ctas_per_cluster == 1
        assert topology.warps_per_cta == 16


@pytest.mark.parametrize("causal", [False, True])
def test_numpy_attention_reference_matches_independent_scalar_gqa(causal):
    rng = np.random.default_rng(7)
    q = rng.standard_normal((1, 3, 4, 2), dtype=np.float32).astype(np.float16)
    k = rng.standard_normal((1, 4, 2, 2), dtype=np.float32).astype(np.float16)
    v = rng.standard_normal((1, 4, 2, 2), dtype=np.float32).astype(np.float16)

    actual = numpy_attention_reference(q, k, v, is_causal=causal)
    expected = _scalar_attention(q, k, v, causal=causal)

    np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=1e-3)
