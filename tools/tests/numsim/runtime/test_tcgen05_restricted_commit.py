"""A-read completion must not publish the rest of a pending MMA."""

import numpy as np
import pytest

from tirx_harness import numsim, racecheck, synccheck
from tests.numsim.runtime.test_tcgen05_i8 import i8_case


@pytest.mark.parametrize(
    "cta_group,tmem_a,multicast", [(1, False, False), (2, False, True), (1, True, True)]
)
def test_restricted_commit_preserves_full_mma_completion(cta_group, tmem_a, multicast, tmp_path):
    for reused_operand in ("a", "b"):
        kernel, args, expected = i8_case(
            cta_group,
            128,
            tmem_a,
            False,
            1,
            1,
            False,
            arch="sm_107a",
            early_reuse=reused_operand,
            restricted_multicast=multicast,
        )
        synccheck(kernel, args).require_clean()
        result = racecheck(kernel, args)
        if reused_operand == "b":
            assert result.findings, "waiting for shared A must not permit overwriting B"
        else:
            result.require_clean()
            actual = (
                numsim.Engine()
                .run(numsim.transpile(kernel, cache_dir=tmp_path), args)
                .outputs["out"]
            )
            np.testing.assert_array_equal(actual, expected)
