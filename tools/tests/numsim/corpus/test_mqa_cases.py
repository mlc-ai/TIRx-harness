from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from tirx_harness.numsim.bindings import prepare_bindings
from tests.numsim.corpus.kernels.deepgemm import (
    FP4_MQA_CONFIGS,
    FP8_MQA_CONFIGS,
    _dense_mqa_reference,
    _e2m1_bits_to_float32,
    _pack_e2m1,
    _pack_e8m0_words,
    _unpack_e2m1,
    _unpack_e8m0_words,
    prepare_fp4_mqa_case,
    prepare_fp8_mqa_case,
)
from tests.numsim.support._tirx_kernels import config_params, load_tirx_kernel
from tirx_harness.numsim.transpiler.frontend import analyze

_DENSE_KERNELS = (
    "deepgemm_sm100_fp4_mqa_logits",
    "deepgemm_sm100_fp8_mqa_logits",
)
_CORPUS = {
    _DENSE_KERNELS[0]: SimpleNamespace(configs=FP4_MQA_CONFIGS, prepare=prepare_fp4_mqa_case),
    _DENSE_KERNELS[1]: SimpleNamespace(configs=FP8_MQA_CONFIGS, prepare=prepare_fp8_mqa_case),
}


def _module(name: str):
    return load_tirx_kernel(name)


def _case_family(module_name: str):
    _module(module_name)
    return _CORPUS[module_name]


def _case(module_name: str, index: int):
    family = _case_family(module_name)
    return family.prepare(**config_params(family.configs[index]))


@pytest.mark.parametrize("module_name", _DENSE_KERNELS)
def test_mqa_numsim_inputs_have_valid_bindings(module_name):
    family = _case_family(module_name)

    for index in range(len(family.configs)):
        prepare_bindings(_case(module_name, index).args)


@pytest.mark.parametrize("module_name", _DENSE_KERNELS)
def test_mqa_shared_pointer_views_keep_their_physical_scope(module_name):
    spec = analyze(_case(module_name, 0).kernel)

    assert not any(
        "declared global view disagrees with shared pointer origin" in item
        for item in spec.unsupported
    )


@pytest.mark.parametrize("module_name", _DENSE_KERNELS)
def test_dense_mqa_numsim_corpus_covers_dense_compressed_and_cooperative(module_name):
    configs = _case_family(module_name).configs

    assert len(configs) == 4
    assert any(not config["compressed_logits"] for config in configs)
    assert any(config["compressed_logits"] for config in configs)
    assert any(not config["disable_cp"] for config in configs)
    assert any(not config["compressed_logits"] and not config["disable_cp"] for config in configs)
    assert {config["logits_dtype"] for config in configs} == {"float32", "bfloat16"}
    assert {config["num_sms"] for config in configs} == {2}


@pytest.mark.parametrize(
    ("module_name", "config_index"),
    [
        (_DENSE_KERNELS[0], 2),
        (_DENSE_KERNELS[1], 1),
    ],
)
def test_mqa_numsim_cases_are_deterministic_and_full_launch(module_name, config_index):
    first = _case(module_name, config_index)
    second = _case(module_name, config_index)

    assert analyze(first.kernel).topology.clusters == 2
    assert analyze(first.kernel).topology.ctas_per_cluster == 1
    assert analyze(first.kernel).topology.warps_per_cta == 12
    np.testing.assert_array_equal(first.reference()["logits"], second.reference()["logits"])
    assert (
        prepare_bindings(first.args).identity_payload()
        == prepare_bindings(second.args).identity_payload()
    )


def test_dense_mqa_numpy_reference_matches_independent_scalar_reference():
    q = np.array(
        [[[1.0, -2.0, 0.5], [0.0, 1.0, 2.0]], [[-1.0, 0.5, 1.5], [2.0, -1.0, 0.0]]],
        dtype=np.float32,
    )
    kv = np.array(
        [[1.0, 0.0, 1.0], [0.5, -1.0, 2.0], [-1.0, 2.0, 0.5], [2.0, 1.0, -1.0]], dtype=np.float32
    )
    weights = np.array([[1.0, -0.25], [0.5, 2.0]], dtype=np.float32)
    starts = np.array([0, 1], dtype=np.int32)
    ends = np.array([3, 4], dtype=np.int32)
    expected = np.full((2, 4), -np.inf, dtype=np.float32)
    for row in range(2):
        for token in range(int(starts[row]), int(ends[row])):
            value = 0.0
            for head in range(2):
                score = sum(float(q[row, head, dim] * kv[token, dim]) for dim in range(3))
                value += max(score, 0.0) * float(weights[row, head])
            expected[row, token] = value

    np.testing.assert_allclose(
        _dense_mqa_reference(q, kv, weights, starts, ends), expected, rtol=0.0, atol=0.0
    )


def test_mqa_low_precision_packing_round_trips_physical_codes():
    codes = np.arange(16, dtype=np.uint8).reshape(2, 8)
    packed = _pack_e2m1(codes)

    np.testing.assert_array_equal(_unpack_e2m1(packed), codes)
    np.testing.assert_array_equal(
        _e2m1_bits_to_float32(np.array([0x2, 0xA], dtype=np.uint8)),
        np.array([1.0, -1.0], dtype=np.float32),
    )
    exponents = np.array([[-2, -1, 0, 1], [1, 0, -1, -2]], dtype=np.int16)
    np.testing.assert_array_equal(
        _unpack_e8m0_words(_pack_e8m0_words(exponents)).astype(np.int16) - 127, exponents
    )


@pytest.mark.parametrize("module_name", _DENSE_KERNELS)
def test_compressed_mqa_comparison_regions_map_prefix_to_kv_range(module_name):
    case = _case(module_name, 1)
    starts = case.args["cu_seq_len_k_start"]
    ends = case.args["cu_seq_len_k_end"]
    logits_stride = int(case.args["logits_stride"])
    seq_len_kv = case.reference()["logits"].size // len(starts)
    regions = case.comparisons["logits"].regions

    for row, (start, end) in enumerate(zip(starts, ends)):
        region = regions[row]
        assert region.actual == (
            slice(row * logits_stride, row * logits_stride + int(end - start)),
        )
        assert region.expected == (
            slice(row * seq_len_kv + int(start), row * seq_len_kv + int(end)),
        )


@pytest.mark.parametrize("module_name", _DENSE_KERNELS)
def test_dense_mqa_comparison_regions_exclude_unspecified_columns(module_name):
    case = _case(module_name, 0)
    starts = case.args["cu_seq_len_k_start"]
    ends = case.args["cu_seq_len_k_end"]
    logits_stride = int(case.args["logits_stride"])
    seq_len_kv = case.reference()["logits"].size // len(starts)
    regions = case.comparisons["logits"].regions

    for row, (start, end) in enumerate(zip(starts, ends)):
        region = regions[row]
        assert region.actual == (
            slice(row * logits_stride + int(start), row * logits_stride + int(end)),
        )
        assert region.expected == (
            slice(row * seq_len_kv + int(start), row * seq_len_kv + int(end)),
        )
