"""Launch masks never execute nonexistent lanes, including after divergence."""

import numpy as np
import pytest

from tests.numsim.support.partial_cta import partial_cta

from tirx_harness import numsim
from tirx_harness.numsim.errors import UnsupportedTIRxError
from tirx_harness.numsim.transpiler.frontend import analyze


@pytest.mark.parametrize("threads", (1, 17, 31, 32, 33, 63, 65))
def test_partial_cta_masks_and_registers(threads, tmp_path):
    function = partial_cta(threads)
    topology = analyze(function).topology
    assert topology.threads_per_cta == threads
    assert topology.warps_per_cta == (threads + 31) // 32
    capacity = topology.warps_per_cta * 32
    initial = np.full((2, capacity, 3), -1, np.int32)
    result = numsim.Engine().run(numsim.transpile(function, cache_dir=tmp_path),
                                 {"output": initial}, outputs=["output"])
    expected = initial.copy()
    indices = np.arange(threads, dtype=np.int32)
    expected[:, :threads, 0] = indices + np.where(indices % 2 == 0, 100, 0)
    expected[:, :threads, 1] = threads
    expected[:, :threads, 2] = indices
    np.testing.assert_array_equal(result.outputs["output"], expected)


@pytest.mark.parametrize("threads", (1, 17, 33))
def test_partial_cta_collectives_fail_closed(threads, tmp_path):
    with pytest.raises(UnsupportedTIRxError, match="partial-warp CTA"):
        numsim.transpile(partial_cta(threads, synchronize=True), cache_dir=tmp_path)
