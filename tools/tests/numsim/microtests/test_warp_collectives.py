from __future__ import annotations

import numpy as np
import pytest

from tests.numsim.microtests.harness import (
    NUMSIM_GPU_MARK,
    require_numsim_gpu,
    run_paired_primfunc,
)
from tests.numsim.runtime.test_ptx_warp_collectives import (
    make_raw_ptx_movmatrix_case,
    raw_ptx_match_float_carriers,
    raw_ptx_match_redux_and_activemask,
    raw_ptx_movmatrix_b16,
    raw_ptx_vote_predicate_carrier,
)


@NUMSIM_GPU_MARK
def test_movmatrix_b16_matches_gpu_and_matrix_transpose_oracle(
    pytestconfig: pytest.Config,
    tmp_path,
):
    require_numsim_gpu(pytestconfig)
    arguments, expected = make_raw_ptx_movmatrix_case()
    result = run_paired_primfunc(
        raw_ptx_movmatrix_b16,
        arguments,
        outputs=("output",),
        cache_dir=tmp_path,
    )

    np.testing.assert_array_equal(result.gpu_outputs["output"], expected)
    np.testing.assert_array_equal(result.numsim_outputs["output"], expected)


@NUMSIM_GPU_MARK
@pytest.mark.parametrize("case", ["match_redux", "match_bits", "ballot_bits"])
def test_warp_register_forms_match_gpu(pytestconfig, tmp_path, case):
    require_numsim_gpu(pytestconfig)
    lanes = np.arange(32, dtype=np.uint32)
    if case == "match_redux":
        kernel = raw_ptx_match_redux_and_activemask
        arguments = {
            "source32": lanes % np.uint32(5),
            "source64": (lanes % np.uint32(3)).astype(np.uint64) << np.uint64(32),
            "partial_mask": np.array([0xFF], dtype=np.uint32),
            "output": np.zeros((32, 13), dtype=np.uint32),
        }
        outputs = ("output",)
    elif case == "match_bits":
        kernel = raw_ptx_match_float_carriers
        arguments = {
            "source32": np.resize(
                np.array([0, 0x80000000, 0x7FC12345, 0x7FC12345], dtype=np.uint32), 32
            ).view(np.float32),
            "source64": np.resize(
                np.array(
                    [0, 0x8000000000000000, 0x7FF8000000001234, 0x7FF8000000001234],
                    dtype=np.uint64,
                ),
                32,
            ).view(np.float64),
            "output32": np.zeros(32, dtype=np.int32),
            "output64": np.zeros(32, dtype=np.uint32),
            "active": np.zeros(32, dtype=np.float32),
        }
        outputs = ("output32", "output64", "active")
    else:
        kernel = raw_ptx_vote_predicate_carrier
        arguments = {
            "predicates": np.where(lanes % 5 == 1, np.uint32(0x80000000), np.uint32(0)),
            "output_u32": np.zeros(32, dtype=np.uint32),
            "output_i32": np.zeros(32, dtype=np.int32),
            "output_f32": np.zeros(32, dtype=np.float32),
        }
        outputs = ("output_u32", "output_i32", "output_f32")
    result = run_paired_primfunc(kernel, arguments, outputs=outputs, cache_dir=tmp_path)
    for name in outputs:
        np.testing.assert_array_equal(
            result.numsim_outputs[name].view(np.uint32),
            result.gpu_outputs[name].view(np.uint32),
        )
