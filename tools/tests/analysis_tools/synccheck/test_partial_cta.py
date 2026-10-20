"""Synccheck executes only the lanes present in a partial CTA launch."""

import numpy as np
import pytest

from tests.numsim.support.partial_cta import partial_cta
from tirx_harness.numsim.checkers import _run_synccheck


@pytest.mark.parametrize("threads", (1, 17, 33))
def test_partial_cta_has_no_ghost_lane_effects(threads, tmp_path):
    capacity = ((threads + 31) // 32) * 32
    report = _run_synccheck(partial_cta(threads),
                           inputs={"output": np.full((2, capacity, 3), -1, np.int32)},
                           cache_dir=tmp_path)
    assert report.verdict == "clean", report.format()
    assert report.native_payload["incomplete"] == []
