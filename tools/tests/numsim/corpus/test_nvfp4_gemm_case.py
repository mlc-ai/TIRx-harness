from __future__ import annotations

import numpy as np
import pytest

from tests.numsim.corpus.kernels.gemm import NVFP4_CONFIGS as NUMSIM_CONFIGS
from tests.numsim.corpus.kernels.gemm import (
    _e2m1_bits_to_float32,
    _float32_to_e2m1_bits,
    _nvfp4_reference,
    _pack_e2m1,
    _pack_sf_128x4,
    _unpack_e2m1,
    _unpack_sf_128x4,
)
from tests.numsim.corpus.kernels.gemm import prepare_nvfp4_case as prepare_numsim_case
from tests.numsim.microtests.harness import NUMSIM_GPU_MARK, require_numsim_gpu
from tests.numsim.support.three_way import run_three_way_case
from tirx_harness.numsim.bindings import _decode_tensor_maps, _tensor_map_base_array
from tirx_harness.numsim.transpiler.frontend import analyze


def _params(config):
    return {key: value for key, value in config.items() if key != "label"}


def test_e2m1_codec_and_nibble_order_are_exact():
    positive = np.arange(8, dtype=np.uint8)
    np.testing.assert_array_equal(
        _e2m1_bits_to_float32(positive),
        np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        _e2m1_bits_to_float32(positive | np.uint8(0x8)),
        -np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        _float32_to_e2m1_bits(np.array([0.75, 1.25, 9.0, -9.0, np.nan], dtype=np.float32)),
        np.array([0x2, 0x2, 0x7, 0xF, 0x7], dtype=np.uint8),
    )

    raw = np.array([[0x1, 0xF, 0x2, 0x8]], dtype=np.uint8)
    packed = _pack_e2m1(raw)
    np.testing.assert_array_equal(packed, np.array([[0xF1, 0x82]], dtype=np.uint8))
    np.testing.assert_array_equal(_unpack_e2m1(packed), raw)


def test_nvfp4_128x4_scale_layout_has_expected_physical_offsets():
    logical = np.zeros((128, 16), dtype=np.uint8)
    logical[0, 0] = 11
    logical[1, 0] = 22
    logical[32, 0] = 33
    logical[0, 4] = 44
    physical = _pack_sf_128x4(logical)
    flat = physical.reshape(-1)

    assert flat[0] == 11
    assert flat[16] == 22
    assert flat[4] == 33
    assert flat[512] == 44
    np.testing.assert_array_equal(_unpack_sf_128x4(physical), logical)


def test_nvfp4_reference_applies_local_e4m3_scales_and_global_alpha():
    A_bits = np.full((128, 64), 0x2, dtype=np.uint8)  # E2M1 +1
    B_bits = np.full((128, 64), 0x2, dtype=np.uint8)
    A_packed = _pack_e2m1(A_bits)
    B_packed = _pack_e2m1(B_bits)
    SFA = _pack_sf_128x4(np.full((128, 4), 0x40, dtype=np.uint8))  # E4M3 2
    SFB = _pack_sf_128x4(np.full((128, 4), 0x30, dtype=np.uint8))  # E4M3 0.5
    alpha = np.array([0.25], dtype=np.float32)

    actual = _nvfp4_reference(A_packed, B_packed, SFA, SFB, alpha)
    expected_f32 = np.full((128, 128), 16.0, dtype=np.float32)
    expected = (expected_f32.view(np.uint32) >> np.uint32(16)).astype(np.uint16)
    np.testing.assert_array_equal(actual, expected)


def test_prepare_nvfp4_numsim_case_is_deterministic_and_full_launch():
    config = NUMSIM_CONFIGS[0]
    assert (config["M"], config["N"], config["K"]) == (256, 256, 256)
    case = prepare_numsim_case(**_params(config))
    repeated = prepare_numsim_case(**_params(config))
    spec = analyze(case.kernel)

    assert spec.topology.clusters == 74
    assert spec.topology.ctas_per_cluster == 2
    assert spec.topology.warps_per_cta == 8
    assert case.subset is None
    assert set(case.outputs) == {"D"}
    assert _tensor_map_base_array(case.args["A_tensor_map"]).shape == (256, 128)
    assert _tensor_map_base_array(case.args["B_tensor_map"]).shape == (256, 128)
    assert _tensor_map_base_array(case.args["SFA_tensor_map"]).shape == (2, 4, 256)
    assert _tensor_map_base_array(case.args["SFB_tensor_map"]).shape == (2, 4, 256)
    assert case.args["alpha"].shape == (1,)
    assert _tensor_map_base_array(case.args["D_tensor_map"]).shape == (256, 256)
    assert {
        name: (
            _decode_tensor_maps(value)[0].dtype
            if name.endswith("_tensor_map")
            else str(value.dtype)
        )
        for name, value in case.args.items()
    } == {
        "A_tensor_map": "uint8",
        "B_tensor_map": "uint8",
        "SFA_tensor_map": "uint16",
        "SFB_tensor_map": "uint16",
        "alpha": "float32",
        "D_tensor_map": "bfloat16",
    }
    np.testing.assert_array_equal(
        _tensor_map_base_array(case.args["A_tensor_map"]),
        _tensor_map_base_array(repeated.args["A_tensor_map"]),
    )
    np.testing.assert_array_equal(
        _tensor_map_base_array(case.args["SFA_tensor_map"]),
        _tensor_map_base_array(repeated.args["SFA_tensor_map"]),
    )
    np.testing.assert_array_equal(case.reference()["D"], repeated.reference()["D"])


@NUMSIM_GPU_MARK
@pytest.mark.parametrize("config", NUMSIM_CONFIGS, ids=lambda config: config["label"])
def test_every_nvfp4_numsim_config_matches_gpu_and_reference(pytestconfig, tmp_path, config):
    require_numsim_gpu(pytestconfig)

    report = run_three_way_case(prepare_numsim_case(**_params(config)), cache_dir=tmp_path)

    report.require_ok()
