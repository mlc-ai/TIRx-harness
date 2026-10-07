"""Racecheck-native test isolation.

The production NumSim default deliberately emits the smaller numerical-only
artifact.  Tests in this directory exercise native analysis entry points, so
their local transpiles opt into the shared analysis-capable artifact without
changing NumSim's public cold-transpile path.
"""

from __future__ import annotations

import pytest

from tirx_harness import numsim


@pytest.fixture(autouse=True)
def analysis_capable_transpile(monkeypatch):
    original = numsim.transpile

    def transpile(*args, **kwargs):
        kwargs.setdefault("_analysis_checker", "racecheck")
        return original(*args, **kwargs)

    monkeypatch.setattr(numsim, "transpile", transpile)
