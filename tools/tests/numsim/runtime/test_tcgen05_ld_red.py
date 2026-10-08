import numpy as np
import pytest

from tests.numsim.microtests.cases.tcgen05_ld_red import ld_red_kernel
from tirx_harness import numsim, racecheck, synccheck


@pytest.mark.parametrize(
    "dtype,maximum,split,absolute,nan",
    [
        ("f32", True, False, False, False),
        ("f32", False, True, True, True),
        ("s32", False, False, False, False),
        ("u32", True, True, False, False),
    ],
)
def test_tcgen_load_reduction_matches_column_oracle(dtype, maximum, split, absolute, nan, tmp_path):
    scalar = {"f32": np.float32, "s32": np.int32, "u32": np.uint32}[dtype]
    source = (np.arange(256).reshape(32, 8) * 37 - 129).astype(scalar)
    if dtype == "f32":
        source[0, :4] = [-0.0, 0.0, -0.0, 0.0]
        source[1, :4] = [np.nan, 1.0, -2.0, 3.0]
        source[2, :4] = np.nan
        source[3, :4] = np.array([1, 2, 3, 4], dtype=np.uint32).view(np.float32)
    loaded = np.concatenate((source[:16, :4], source[:16, 4:]), axis=0) if split else source[:, :4]
    values = np.abs(loaded) if absolute else loaded
    op = (np.maximum if maximum else np.minimum) if nan else (np.fmax if maximum else np.fmin)
    expected = op.reduce(values, axis=1)
    if dtype == "f32":
        # PTX max/min select +0/-0 respectively when both signs are present.
        zero = (values == 0).all(axis=1)
        expected[zero] = 0.0 if maximum or absolute else -0.0
    result = numsim.Engine().run(
        numsim.transpile(ld_red_kernel(dtype, maximum, split, absolute, nan), cache_dir=tmp_path),
        {"source": source, "output": np.zeros((32, 5), dtype=scalar)},
        outputs=("output",),
    )
    actual = result.outputs["output"]
    np.testing.assert_array_equal(actual[:, :4], loaded)
    np.testing.assert_array_equal(actual[:, 4], expected)
    if dtype == "f32":
        np.testing.assert_array_equal(np.signbit(actual[:, 4][zero]), np.signbit(expected[zero]))


def test_tcgen_load_reduction_preserves_the_store_wait_contract(monkeypatch, tmp_path):
    monkeypatch.setenv("NUMSIM_CACHE_DIR", str(tmp_path))
    args = {
        "source": np.arange(256, dtype=np.float32).reshape(32, 8),
        "output": np.zeros((32, 5), dtype=np.float32),
    }
    for checker in (synccheck, racecheck):
        checker(ld_red_kernel(), args).require_clean()
    report = racecheck(ld_red_kernel(wait_st=False), args)
    assert report.verdict == "error"
    assert any(f.status == "error" and f.details["access_pair"] == "write_read" for f in report.findings)
