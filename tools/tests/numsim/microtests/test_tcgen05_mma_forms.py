from __future__ import annotations

import numpy as np
import pytest

from tests.numsim.microtests.cases.tcgen05_mma_forms import (
    TCGEN05_MMA_FORM_CASES,
    Tcgen05MmaFormCase,
    e5m2_layout_f_reference,
    f16_destination_reference,
    f16_kind_destination_reference,
    make_raw_e4m3_e5m2_arguments,
    make_raw_e5m2_arguments,
    make_raw_f16_destination_arguments,
    make_raw_f16_kind_destination_arguments,
    raw_e4m3_e5m2_ss_m64_layout_f_valid_descriptor,
    raw_e5m2_e4m3_f16_d_ss_m128_layout_d,
    raw_e5m2_ss_m64_layout_f_valid_descriptor,
    raw_f16_f16_d_ss_m128_layout_d,
)
from tests.numsim.microtests.harness import (
    NUMSIM_GPU_MARK,
    require_numsim_gpu,
    run_paired_primfunc,
)


@NUMSIM_GPU_MARK
@pytest.mark.parametrize("case", TCGEN05_MMA_FORM_CASES, ids=lambda case: case.name)
def test_tcgen05_mma_form_matches_gpu(
    case: Tcgen05MmaFormCase,
    pytestconfig: pytest.Config,
    tmp_path,
):
    require_numsim_gpu(pytestconfig)
    run_paired_primfunc(
        case.prim_func,
        case.make_arguments(),
        outputs=case.outputs,
        cache_dir=tmp_path,
        max_ulp=case.max_ulp,
    )


@NUMSIM_GPU_MARK
@pytest.mark.parametrize(
    ("prim_func", "make_arguments", "a_dtype", "b_dtype"),
    [
        (
            raw_e5m2_ss_m64_layout_f_valid_descriptor,
            make_raw_e5m2_arguments,
            "float8_e5m2",
            "float8_e5m2",
        ),
        (
            raw_e4m3_e5m2_ss_m64_layout_f_valid_descriptor,
            make_raw_e4m3_e5m2_arguments,
            "float8_e4m3fn",
            "float8_e5m2",
        ),
    ],
    ids=("e5m2_e5m2", "e4m3_e5m2"),
)
def test_raw_f8f6f4_e5m2_operands_use_the_e5m2_decoder(
    prim_func,
    make_arguments,
    a_dtype,
    b_dtype,
    pytestconfig: pytest.Config,
    tmp_path,
):
    """E5M2 has no typed-path oracle, so pin it against hardware and numpy.

    The negative controls matter more than the positive one here.

    E4M3 and E5M2 are both eight bits, so a lowering that reached the E4M3
    decoder would still produce plausible finite numbers; reading the same
    bytes as E4M3 must disagree with the GPU.

    The payloads also include the three E5M2 subnormals, which take a separate
    decoder branch. The second control shows the hardware does not flush them:
    a flush-to-zero reading disagrees with the GPU in most output elements, so
    matching the exact reading is a real measurement rather than a coincidence.
    """

    require_numsim_gpu(pytestconfig)
    arguments = make_arguments()
    result = run_paired_primfunc(
        prim_func,
        arguments,
        outputs=("output",),
        cache_dir=tmp_path,
    )

    expected = e5m2_layout_f_reference(
        arguments, a_dtype=a_dtype, b_dtype=b_dtype, decode_as_e4m3=False
    )
    np.testing.assert_array_equal(result.gpu_outputs["output"], expected)
    np.testing.assert_array_equal(result.numsim_outputs["output"], expected)

    wrong_decoder = e5m2_layout_f_reference(
        arguments, a_dtype=a_dtype, b_dtype=b_dtype, decode_as_e4m3=True
    )
    assert not np.array_equal(wrong_decoder, expected)
    assert not np.array_equal(result.gpu_outputs["output"], wrong_decoder)

    flushed = e5m2_layout_f_reference(
        arguments, a_dtype=a_dtype, b_dtype=b_dtype, flush_subnormals=True
    )
    assert np.count_nonzero(flushed != expected) > expected.size // 2
    assert not np.array_equal(result.gpu_outputs["output"], flushed)


@NUMSIM_GPU_MARK
def test_raw_f8f6f4_float16_destination_converts_once_after_an_f32_accumulation(
    pytestconfig: pytest.Config,
    tmp_path,
):
    """Pin the measured `.f16` destination behaviour, both halves of the word.

    The addend is 1024.0, where binary16 has a ULP of 1, and 31 of the 32
    products are 0.25, so the exact sum 1031.75 is *not* representable in
    binary16 and every hypothesis lands on a different word:

      * f32 accumulate, round-to-nearest on store -> 1032 (0x6408)
      * f32 accumulate, truncate on store         -> 1031 (0x6407)
      * round to binary16 between K steps         -> 1024 (0x6400)

    The seeded upper half is 0xBEEF, so a store that preserved it instead of
    zeroing it also fails. What this does *not* separate is an f32 reduction
    from any wider or differently-associated one that agrees with f32 here; the
    PTX ISA is silent on tcgen05 accumulation precision, so the model claims
    only what these bytes show.
    """

    require_numsim_gpu(pytestconfig)
    arguments = make_raw_f16_destination_arguments()
    result = run_paired_primfunc(
        raw_e5m2_e4m3_f16_d_ss_m128_layout_d,
        arguments,
        outputs=("output",),
        cache_dir=tmp_path,
    )

    # The store's rounding is only under test while the exact sum needs it.
    exact = np.float32(1024.0) + np.float32(31) * np.float32(0.25)
    assert np.float32(np.float16(exact)) != exact

    expected = f16_destination_reference(arguments)
    assert np.unique(expected).tolist() == [0x6408]  # binary16 1032.0, zero upper half
    np.testing.assert_array_equal(result.gpu_outputs["output"], expected)
    np.testing.assert_array_equal(result.numsim_outputs["output"], expected)

    per_step = f16_destination_reference(arguments, per_step_f16=True)
    assert np.unique(per_step).tolist() == [0x6400]  # binary16 1024.0
    assert not np.array_equal(result.gpu_outputs["output"], per_step)

    truncating_store = np.full_like(expected, 0x6407)  # binary16 1031.0
    assert not np.array_equal(result.gpu_outputs["output"], truncating_store)

    preserved_upper_half = expected | np.uint32(0xBEEF << 16)
    assert not np.array_equal(result.gpu_outputs["output"], preserved_upper_half)


@NUMSIM_GPU_MARK
def test_raw_f16_kind_float16_destination_converts_once_after_an_f32_accumulation(
    pytestconfig: pytest.Config,
    tmp_path,
):
    """The same probe as the `.kind::f8f6f4` case, now under `.kind::f16`.

    PTX ISA 9.7.17.4.2 Table 45 allows D=F16 for `.kind::f16` (bits 4-5 = 0),
    and NumSim models it with the destination codec it measured for
    `.kind::f8f6f4`. This test is what holds that reuse honest: the addend is
    1024.0 and 15 of the 16 products are 0.25, so the exact sum 1027.75 is not
    representable in binary16 and each hypothesis lands on a different word:

      * f32 accumulate, round-to-nearest on store -> 1028 (0x6404)
      * f32 accumulate, truncate on store         -> 1027 (0x6403)
      * round to binary16 between K steps         -> 1024 (0x6400)

    The seeded upper half is 0xBEEF, so a store that preserved it instead of
    zeroing it also fails.
    """

    require_numsim_gpu(pytestconfig)
    arguments = make_raw_f16_kind_destination_arguments()
    result = run_paired_primfunc(
        raw_f16_f16_d_ss_m128_layout_d,
        arguments,
        outputs=("output",),
        cache_dir=tmp_path,
    )

    # The store's rounding is only under test while the exact sum needs it.
    exact = np.float32(1024.0) + np.float32(15) * np.float32(0.25)
    assert np.float32(np.float16(exact)) != exact

    expected = f16_kind_destination_reference(arguments)
    assert np.unique(expected).tolist() == [0x6404]  # binary16 1028.0, zero upper half
    np.testing.assert_array_equal(result.gpu_outputs["output"], expected)
    np.testing.assert_array_equal(result.numsim_outputs["output"], expected)

    per_step = f16_kind_destination_reference(arguments, per_step_f16=True)
    assert np.unique(per_step).tolist() == [0x6400]  # binary16 1024.0
    assert not np.array_equal(result.gpu_outputs["output"], per_step)

    truncating_store = np.full_like(expected, 0x6403)  # binary16 1027.0
    assert not np.array_equal(result.gpu_outputs["output"], truncating_store)

    preserved_upper_half = expected | np.uint32(0xBEEF << 16)
    assert not np.array_equal(result.gpu_outputs["output"], preserved_upper_half)
