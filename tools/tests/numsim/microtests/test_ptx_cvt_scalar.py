"""Scalar PTX ``cvt`` semantics, checked against recorded and live hardware.

The recorded test runs the whole TIRx -> transpile -> execute path with no GPU
and compares every destination payload with the goldens measured on the device
named in ``SCALAR_CVT_GPU``.  The paired test re-measures the same kernels on
the live GPU, so a driver or architecture change that moves the semantics is
caught rather than silently blessed.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.numsim.microtests.cases.ptx_cvt_scalar import (
    SCALAR_CVT_COLUMNS,
    SCALAR_CVT_KERNELS,
    scalar_cvt_arguments,
)
from tests.numsim.microtests.cases.ptx_cvt_scalar_goldens import SCALAR_CVT_GOLDENS
from tests.numsim.microtests.harness import (
    NUMSIM_GPU_MARK,
    require_numsim_gpu,
    run_paired_primfunc,
)
from tirx_harness import numsim

_RAW_VIEW = {
    np.dtype(np.int8): np.uint8,
    np.dtype(np.uint8): np.uint8,
    np.dtype(np.int16): np.uint16,
    np.dtype(np.uint16): np.uint16,
    np.dtype(np.int32): np.uint32,
    np.dtype(np.uint32): np.uint32,
    np.dtype(np.int64): np.uint64,
    np.dtype(np.uint64): np.uint64,
    np.dtype(np.float32): np.uint32,
    np.dtype(np.float64): np.uint64,
}


def _payloads(array: np.ndarray, column: int) -> np.ndarray:
    """Raw destination bits, so NaN payloads and signed zeros compare exactly."""

    return array.view(_RAW_VIEW[array.dtype])[:, column].astype(np.uint64)


@pytest.mark.parametrize("name", sorted(SCALAR_CVT_KERNELS), ids=str)
def test_scalar_cvt_matches_recorded_gpu_goldens(name: str, tmp_path):
    arguments = scalar_cvt_arguments(name)
    outputs = tuple(SCALAR_CVT_COLUMNS[name])
    module = numsim.transpile(SCALAR_CVT_KERNELS[name], cache_dir=tmp_path)
    result = numsim.Engine().run(module, arguments, outputs=outputs)

    checked = 0
    for buffer, spellings in SCALAR_CVT_COLUMNS[name].items():
        observed = np.asarray(result.outputs[buffer])
        for column, spelling in enumerate(spellings):
            expected = np.asarray(SCALAR_CVT_GOLDENS[spelling], dtype=np.uint64)
            np.testing.assert_array_equal(
                _payloads(observed, column),
                expected,
                err_msg=f"{spelling} disagrees with the recorded GPU result",
            )
            checked += 1
    assert checked == sum(len(spellings) for spellings in SCALAR_CVT_COLUMNS[name].values())


def test_every_recorded_spelling_is_exercised_by_a_kernel():
    covered = {
        spelling
        for columns in SCALAR_CVT_COLUMNS.values()
        for spellings in columns.values()
        for spelling in spellings
    }
    assert covered == set(SCALAR_CVT_GOLDENS)


@NUMSIM_GPU_MARK
@pytest.mark.parametrize("name", sorted(SCALAR_CVT_KERNELS), ids=str)
def test_scalar_cvt_matches_live_gpu(name: str, pytestconfig: pytest.Config, tmp_path):
    require_numsim_gpu(pytestconfig)
    run_paired_primfunc(
        SCALAR_CVT_KERNELS[name],
        scalar_cvt_arguments(name),
        outputs=tuple(SCALAR_CVT_COLUMNS[name]),
        cache_dir=tmp_path,
    )
