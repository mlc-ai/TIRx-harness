from __future__ import annotations

import builtins

import numpy as np
import pytest
import torch

from tests.numsim.corpus.kernels.gemm import (
    FP8_1D1D_NUM_SMS,
    FP8_BLOCKWISE_CONFIGS as NUMSIM_CONFIGS,
)
from tests.numsim.corpus.kernels.gemm import (
    _e4m3fn_bits_to_float32,
    _e8m0_bits_from_exponents,
    _e8m0_bits_to_float32,
    _float32_to_e4m3fn_bits,
    _fp8_blockwise_reference,
    _pack_e8m0_scales,
    _unpack_e8m0_scales,
)
from tests.numsim.corpus.kernels.gemm import (
    prepare_fp8_blockwise_case as prepare_numsim_case,
)
from tests.numsim.microtests.harness import NUMSIM_GPU_MARK, require_numsim_gpu
from tests.numsim.support._tirx_kernels import load_tirx_kernel
from tests.numsim.support.three_way import run_three_way_case
from tirx_harness.numsim.bindings import _decode_tensor_maps, _tensor_map_base_array
from tirx_harness.numsim.transpiler.frontend import analyze


def _params(config):
    return {key: value for key, value in config.items() if key != "label"}


def _descriptor_semantics(descriptor: np.ndarray) -> tuple[object, ...]:
    decoded = _decode_tensor_maps(descriptor)[0]
    return (
        decoded.required_byte_len,
        decoded.global_shape,
        decoded.global_strides,
        decoded.box_shape,
        decoded.element_strides,
        decoded.dtype,
        decoded.fp4_shared_layout,
        decoded.swizzle,
        decoded.fill_mode,
    )


def _fp8_kernel_module():
    try:
        return load_tirx_kernel("deepgemm_sm100_fp8_gemm_1d1d")
    except ModuleNotFoundError as error:
        if error.name and (error.name == "deep_gemm" or error.name.startswith("deep_gemm.")):
            pytest.skip("FP8 blockwise kernel requires the optional deep_gemm package")
        raise


def test_fp8_numsim_configs_cover_both_deepgemm_layout_branches():
    module = _fp8_kernel_module()
    assert [(config["M"], config["N"], config["K"]) for config in NUMSIM_CONFIGS] == [
        (16, 256, 512),
        (512, 608, 512),
    ]
    actual = {
        module._spec_for(
            {
                "M": config["M"],
                "N": config["N"],
                "K": config["K"],
                "num_sms": 2,
            }
        ).swap_ab
        for config in NUMSIM_CONFIGS
    }
    assert actual == {False, True}
    assert {config["expected_swap_ab"] for config in NUMSIM_CONFIGS} == actual


def test_e4m3fn_codec_handles_subnormals_rounding_saturation_and_nan():
    bits = np.array([0x00, 0x01, 0x07, 0x08, 0x38, 0x3C, 0x7E, 0x80, 0xFE, 0x7F])
    decoded = _e4m3fn_bits_to_float32(bits)
    np.testing.assert_array_equal(
        decoded[:-1],
        np.array([0.0, 2**-9, 7 * 2**-9, 2**-6, 1.0, 1.5, 448.0, -0.0, -448.0], dtype=np.float32),
    )
    assert np.isnan(decoded[-1])

    values = np.array([1.0625, 1.1875, 1000.0, -1000.0, np.nan], dtype=np.float32)
    np.testing.assert_array_equal(
        _float32_to_e4m3fn_bits(values), np.array([0x38, 0x3A, 0x7E, 0xFE, 0x7F], dtype=np.uint8)
    )


def test_e8m0_scale_encoding_and_uint32_packing_are_exact():
    exponents = np.array([[-1, 0, 1, 2], [3, -2, 4, -3]], dtype=np.int16)
    codes = _e8m0_bits_from_exponents(exponents)
    packed = _pack_e8m0_scales(codes)

    assert packed.shape == (1, 2)
    assert int(packed[0, 0]) == 0x81807F7E
    assert int(packed[0, 1]) == 0x7C837D82
    np.testing.assert_array_equal(_unpack_e8m0_scales(packed, 4), codes)
    np.testing.assert_array_equal(_e8m0_bits_to_float32(codes), np.exp2(exponents))
    endpoints = _e8m0_bits_to_float32(np.array([0x00, 0x7F, 0xFE, 0xFF]))
    np.testing.assert_array_equal(
        endpoints[:3], np.array([2.0**-127, 1.0, 2.0**127], dtype=np.float32)
    )
    assert np.isnan(endpoints[3])
    with pytest.raises(ValueError, match="exponent"):
        _e8m0_bits_from_exponents([-128])


def test_fp8_reference_applies_one_e8m0_scale_per_128_k_values():
    one = int(_float32_to_e4m3fn_bits(np.array([1.0], dtype=np.float32))[0])
    A = np.full((1, 512), one, dtype=np.uint8)
    B = np.full((1, 512), one, dtype=np.uint8)
    SFA = _pack_e8m0_scales(_e8m0_bits_from_exponents([[0, 1, -1, 2]]))
    SFB = _pack_e8m0_scales(_e8m0_bits_from_exponents([[0, -1, 1, -2]]))

    expected = np.float32(128 * (1 * 1 + 2 * 0.5 + 0.5 * 2 + 4 * 0.25))
    actual_bits = _fp8_blockwise_reference(A, B, SFA, SFB)
    expected_bits = np.asarray(expected, dtype=np.float32).view(np.uint32) >> np.uint32(16)
    assert int(actual_bits[0, 0]) == int(expected_bits)


@pytest.mark.parametrize("config", NUMSIM_CONFIGS, ids=lambda config: config["label"])
def test_prepare_fp8_numsim_case_is_cpu_only_deterministic_and_full_launch(config, monkeypatch):
    module = _fp8_kernel_module()
    M, N, K = config["M"], config["N"], config["K"]
    kernel_spec = module._spec_for({"M": M, "N": N, "K": K, "num_sms": FP8_1D1D_NUM_SMS})
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "deep_gemm" or name.startswith("deep_gemm."):
            raise AssertionError("NumSim case imported DeepGEMM")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    monkeypatch.setattr(
        torch.cuda,
        "_lazy_init",
        lambda: (_ for _ in ()).throw(AssertionError("NumSim case initialized CUDA")),
    )
    case = prepare_numsim_case(**_params(config))
    repeated = prepare_numsim_case(**_params(config))
    spec = analyze(case.kernel)

    assert spec.topology.clusters == FP8_1D1D_NUM_SMS // spec.topology.ctas_per_cluster
    assert spec.topology.ctas_per_cluster == 2
    assert spec.topology.warps_per_cta == 8
    assert case.subset is None
    assert case.outputs == {"D": "tensor_map_cd"}
    assert set(case.args) == {
        "grouped_layout",
        "grouped_len",
        "shape_m",
        "shape_n",
        "shape_k",
        "tensor_map_a",
        "tensor_map_b",
        "tensor_map_sfa",
        "tensor_map_sfb",
        "tensor_map_cd",
    }
    grouped_layout = case.args["grouped_layout"]
    assert isinstance(grouped_layout, np.ndarray)
    np.testing.assert_array_equal(grouped_layout, np.zeros((1,), dtype=np.int32))
    assert {
        name: int(case.args[name]) for name in ("grouped_len", "shape_m", "shape_n", "shape_k")
    } == {"grouped_len": 1, "shape_m": M, "shape_n": N, "shape_k": K}

    tensor_map_names = (
        "tensor_map_a",
        "tensor_map_b",
        "tensor_map_sfa",
        "tensor_map_sfb",
        "tensor_map_cd",
    )
    for name in tensor_map_names:
        descriptor = case.args[name]
        assert isinstance(descriptor, np.ndarray)
        assert descriptor.dtype == np.uint8
        assert descriptor.shape == (128,)
        assert _descriptor_semantics(descriptor) == _descriptor_semantics(repeated.args[name])
        np.testing.assert_array_equal(
            _tensor_map_base_array(descriptor),
            _tensor_map_base_array(repeated.args[name]),
        )
    np.testing.assert_array_equal(case.reference()["D"], repeated.reference()["D"])


@NUMSIM_GPU_MARK
def test_fp8_blockwise_kernel_selects_each_128_k_scale_block(pytestconfig, tmp_path):
    require_numsim_gpu(pytestconfig)
    _fp8_kernel_module()
    case = prepare_numsim_case(
        M=16,
        N=256,
        K=512,
        expected_swap_ab=True,
        seed=2800,
    )

    report = run_three_way_case(case, cache_dir=tmp_path)

    report.require_ok()
