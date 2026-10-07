"""Sparse B16 uses selected 2:4 terms, common MMA lifetimes and D codecs."""

import numpy as np
import pytest

from tests.numsim.runtime.test_tcgen05_ti16 import sparse_b16_case, ti16_kernel
from tests.numsim.support.execution import assert_rejected, run_checked
from tirx_harness import racecheck

SPARSE_B16_CASES = [
    (1, 64, False, 0, 0, False, False, False),
    (1, 128, True, 1, 1, False, False, False),
    (2, 128, True, 0, 0, True, False, False),
    (2, 256, False, 0, 0, True, False, False),
    (1, 128, False, 1, 1, False, True, True),
    (2, 256, True, 0, 0, False, False, True),
]


def encode_b16(values, bf16):
    return (
        (values.view(np.uint32) >> 16).astype(np.uint16)
        if bf16
        else values.astype(np.float16).view(np.uint16)
    )


def sparse_float_case(cta, m, tmem, af, bf, half, ta, tb, *, selector=1, early_reuse=None):
    _, args, dot = sparse_b16_case(m, tmem, False, cta_group=cta)
    a = -args["a"].astype(np.float32) / 8
    b = args["b"].astype(np.float32) / 16
    args["a"], args["b"] = encode_b16(a, af), encode_b16(b, bf)
    seed = (np.arange(m * (16 * cta)).reshape(m, -1) % 7 / 16).astype(np.float32)
    expected = dot.astype(np.float32) * (-1 / 128) + seed
    args["seed"] = (
        (seed.astype(np.float16).view(np.uint16).astype(np.uint32) | np.uint32(0xBEEF0000)).view(
            np.int32
        )
        if half
        else seed.view(np.int32)
    )
    expected = (
        expected.astype(np.float16).view(np.uint16).astype(np.int32)
        if half
        else expected.view(np.int32)
    )
    if cta == 2:
        expected[m // 2 + 1] = args["seed"][m // 2 + 1]
    if selector == 0:
        args["metadata"][:, :, 0] = args["metadata"][:, :, 1]
        args["metadata"][:, :, 1] = 0
    kernel = ti16_kernel(
        True,
        tmem,
        cta_group=cta,
        m=m,
        kind="f16",
        sparse=True,
        a_format=af,
        b_format=bf,
        half_accumulator=half,
        transpose_a=ta,
        transpose_b=tb,
        sparsity_selector=selector,
        early_reuse=early_reuse,
        collectors=".collector::a::discard.collector::b::discard",
        arch="sm_107a",
    )
    return kernel, args, expected


@pytest.mark.parametrize("case", SPARSE_B16_CASES)
def test_sparse_b16_layouts_and_codecs(case, tmp_path):
    kernel, args, expected = sparse_float_case(*case, selector=int(not case[2]))
    result = run_checked(kernel, args, cache_dir=tmp_path)
    np.testing.assert_array_equal(result.outputs["out"], expected)


def test_sparse_b16_metadata_and_b_lifetimes():
    for resource in ("b", "lookup"):
        kernel, args, _ = sparse_float_case(*SPARSE_B16_CASES[0], early_reuse=resource)
        for disabled in (True, False):
            args["zero_mask"][0] = np.uint64(1 << 63 if disabled else 0)
            report = racecheck(kernel, args)
            assert report.verdict == ("clean" if disabled else "error"), report.to_dict()


def test_sparse_b16_invalid_metadata(tmp_path):
    kernel, args, _ = sparse_float_case(*SPARSE_B16_CASES[0])
    args["metadata"][:, :, 1] = 0x33333333
    assert_rejected(kernel, args, "not a defined index pair", cache_dir=tmp_path)
