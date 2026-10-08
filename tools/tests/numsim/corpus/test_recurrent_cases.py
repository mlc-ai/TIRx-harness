from __future__ import annotations

import numpy as np

from tests.numsim.corpus.kernels.recurrent import (
    gdn_numpy_reference,
    prepare_native_kda_fixed_case,
    prepare_native_kda_forward_case,
    prepare_gdn_prefill_case,
)
from tirx_harness.numsim import run_case
from tirx_harness.numsim.bindings import _decode_tensor_maps, _tensor_map_base_array
from tirx_harness.numsim.transpiler.frontend import analyze


def _bfloat16_bits(value: float) -> np.uint16:
    bits = np.asarray([value], dtype=np.float32).view(np.uint32)[0]
    return np.uint16(bits >> np.uint32(16))


def _descriptor_metadata(array: np.ndarray) -> tuple[tuple[object, ...], ...]:
    return tuple(
        (
            descriptor.byte_offset,
            descriptor.required_byte_len,
            descriptor.global_shape,
            descriptor.global_strides,
            descriptor.box_shape,
            descriptor.element_strides,
            descriptor.dtype,
            descriptor.fp4_shared_layout,
            descriptor.swizzle,
            descriptor.fill_mode,
        )
        for descriptor in _decode_tensor_maps(array)
    )


def _assert_same_arguments(actual: dict[str, object], expected: dict[str, object]) -> None:
    assert actual.keys() == expected.keys()
    for name, value in actual.items():
        repeated = expected[name]
        if not isinstance(value, np.ndarray):
            assert value == repeated
            continue
        assert isinstance(repeated, np.ndarray)
        descriptors = _decode_tensor_maps(value)
        if not descriptors:
            np.testing.assert_array_equal(value, repeated)
            continue
        assert _descriptor_metadata(value) == _descriptor_metadata(repeated)
        for descriptor in descriptors:
            start = descriptor.byte_offset
            np.testing.assert_array_equal(
                _tensor_map_base_array(value.view(np.uint8).reshape(-1)[start : start + 128]),
                _tensor_map_base_array(repeated.view(np.uint8).reshape(-1)[start : start + 128]),
            )


def test_native_kda_case_is_one_complete_head_cta() -> None:
    case = prepare_native_kda_forward_case()
    topology = analyze(case.kernel).topology

    assert topology.clusters == 64
    assert topology.ctas_per_cluster == 1
    assert topology.warps_per_cta == 20
    assert topology.warp_count == 64 * 20
    assert case.subset is None
    assert "num_ctas" not in case.args
    assert case.reference()["output"].shape == (2, 128, 64)
    assert "final_state" in case.outputs
    assert case.reference()["final_state"].shape == (2 * 64 * 128 * 128,)
    [region] = case.comparisons["output"].regions
    assert region.actual == (slice(0, 2), slice(None), slice(None))
    assert region.expected == (slice(None), slice(None), slice(None))


def test_native_kda_fixed_route_matches_independent_oracle() -> None:
    case = prepare_native_kda_fixed_case()
    assert len(case.kernel) == 2
    assert analyze(case.kernel[1]).topology.clusters == 64
    report = run_case(case)
    report.require_ok()


def test_gdn_prefill_case_is_deterministic_and_full_launch() -> None:
    case = prepare_gdn_prefill_case()
    repeated = prepare_gdn_prefill_case()
    topology = analyze(case.kernel).topology

    assert topology.clusters == 1
    assert topology.ctas_per_cluster == 1
    assert topology.warps_per_cta == 12
    assert case.outputs == ("o", "final_state")
    _assert_same_arguments(case.args, repeated.args)
    for name in case.outputs:
        np.testing.assert_array_equal(case.reference()[name], repeated.reference()[name])


def test_gdn_oracle_detects_value_corruption() -> None:
    case = prepare_gdn_prefill_case()
    q = case.args["q"].reshape(1, 2, 128)
    k = case.args["k"].reshape(1, 2, 128)
    v = case.args["v"].reshape(1, 8, 128)
    gate = case.args["gate"].reshape(1, 8)
    beta = case.args["beta"].reshape(1, 8)
    initial_state = case.args["initial_state"].reshape(1, 8, 128, 128)
    expected, expected_state = gdn_numpy_reference(
        q,
        k,
        v,
        gate,
        beta,
        initial_state,
        seq_lens=(1,),
        scale=case.args["scale"],
    )

    corrupted_v = v.copy()
    corrupted_v[0, 0, 0] = np.float16(corrupted_v[0, 0, 0] + np.float16(1.0))
    corrupted, corrupted_state = gdn_numpy_reference(
        q,
        k,
        corrupted_v,
        gate,
        beta,
        initial_state,
        seq_lens=(1,),
        scale=case.args["scale"],
    )

    assert not np.array_equal(corrupted, expected)
    assert not np.array_equal(corrupted_state, expected_state)
