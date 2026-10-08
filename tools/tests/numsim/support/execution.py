"""The ordinary clean-checker path used by focused numerical tests."""

import pytest

from tirx_harness import numsim, racecheck, synccheck


def run_checked(kernel, inputs, *, cache_dir=None, outputs=None):
    """Keep each test's inputs and oracle separate from the shared tool invocation."""
    for checker in (synccheck, racecheck):
        checker(kernel, inputs).require_clean()
    return numsim.Engine().run(
        numsim.transpile(kernel, cache_dir=cache_dir),
        inputs,
        outputs=outputs,
    )


def assert_rejected(kernel, inputs, reason, *, verdict="error", cache_dir=None):
    """Check the same rejected invocation through both checkers and NumSim."""
    reports = [checker(kernel, inputs) for checker in (synccheck, racecheck)]
    for report in reports:
        assert report.verdict == verdict, report.format()
        assert reason in report.format(), report.format()
    with pytest.raises(numsim.NumSimExecutionError, match=reason):
        numsim.Engine().run(numsim.transpile(kernel, cache_dir=cache_dir), inputs)
    return reports
