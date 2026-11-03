"""The ordinary clean-checker path used by focused numerical tests."""

import json

import pytest

from tirx_harness import numsim, racecheck, synccheck
from tirx_harness.numsim.checker_runner import _specialize
from tvm_tirx import ir, tirx


def specialize(kernel, bindings):
    """``kernel`` with each bound parameter substituted, as TVM's Function.specialize did."""
    return _specialize(
        kernel,
        {
            parameter: value
            if isinstance(value, ir.Expr)
            else tirx.const(
                value.item() if hasattr(value, "item") else value, str(parameter.ty.dtype)
            )
            for parameter, value in bindings.items()
        },
    )


def with_cuda_arch(kernel, arch):
    """``kernel`` compiled for the CUDA architecture ``arch``."""
    return kernel.with_attr("backend_config", json.dumps({"cuda": {"arch": arch}}))


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
