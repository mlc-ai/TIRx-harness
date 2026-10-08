from __future__ import annotations

import numpy as np
import pytest

from tests.numsim.integration.test_instruction_profile import (
    assert_executed_branch,
    profiled_branch,
)
from tirx_harness import numsim


@pytest.mark.parametrize("checker", ["synccheck", "racecheck"])
@pytest.mark.parametrize("profiled", [False, True])
def test_profiled_checker_preserves_verdict_and_observes_only_executed_branch(
    monkeypatch, tmp_path, checker, profiled
):
    monkeypatch.setenv("NUMSIM_PROFILE", "1" if profiled else "0")
    module = numsim.transpile(
        profiled_branch,
        cache_dir=tmp_path,
        _analysis_capable=True,
        _analysis_checker=checker,
    )
    for choose_add in (0, 1):
        result = getattr(numsim.Engine(max_workers=2), f"run_{checker}_phase")(
            module,
            {
                "source": np.arange(64, dtype=np.uint32) + 10,
                "output": np.zeros(64, dtype=np.uint32),
                "choose_add": choose_add,
            },
        )
        payload = result.to_dict()
        assert result.verdict == "clean", payload
        assert payload["incomplete"] == []
        assert payload["execution_error"] is None
        if profiled:
            assert_executed_branch(
                module.spec.kernels[0],
                payload["stats"]["executed_instruction_variants"],
                choose_add,
            )
        else:
            assert "executed_instruction_variants" not in payload["stats"]
