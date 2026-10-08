from __future__ import annotations

import numpy as np
import pytest

from tirx_harness.numsim import checkers

from tests.numsim.runtime.test_raw_tcgen_codegen import raw_tcgen_mma_bf16_ss


@pytest.mark.parametrize(
    "run",
    [checkers._run_synccheck, checkers._run_racecheck],
    ids=["synccheck", "racecheck"],
)
def test_checker_follows_matrix_descriptor_through_raw_shared_table(run, tmp_path):
    report = run(
        raw_tcgen_mma_bf16_ss,
        inputs={
            "a_physical": np.zeros(4096, dtype=np.uint8),
            "b_physical": np.zeros(1024, dtype=np.uint8),
            "output": np.zeros((64, 8), dtype=np.float32),
            "output_ws": np.zeros((64, 8), dtype=np.float32),
        },
        cache_dir=tmp_path,
        max_workers=1,
    )

    report.require_clean()
    if run is checkers._run_racecheck:
        assert report.to_dict()["native"]["access_count"] > 0
