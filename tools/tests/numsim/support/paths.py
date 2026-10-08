"""Stable repository paths shared by the external NumSim test tree."""

from pathlib import Path

import tirx_harness.numsim as numsim_package


NUMSIM_TEST_ROOT = Path(__file__).resolve().parents[1]
TOOLS_ROOT = NUMSIM_TEST_ROOT.parents[1]
REPO_ROOT = TOOLS_ROOT.parent
NUMSIM_PACKAGE_ROOT = Path(numsim_package.__file__).resolve().parent
ENGINE_ROOT = NUMSIM_PACKAGE_ROOT / "engine-rs"


def find_test_file(filename: str) -> Path:
    """Resolve a unique NumSim test filename below the layered test tree."""

    matches = tuple(NUMSIM_TEST_ROOT.rglob(filename))
    if len(matches) != 1:
        raise ValueError(f"expected one NumSim test file {filename!r}, found {matches!r}")
    return matches[0]


__all__ = [
    "ENGINE_ROOT",
    "NUMSIM_PACKAGE_ROOT",
    "NUMSIM_TEST_ROOT",
    "REPO_ROOT",
    "TOOLS_ROOT",
    "find_test_file",
]
