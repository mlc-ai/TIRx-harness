from __future__ import annotations

import json
import os
import subprocess
import sys

from tests.numsim.support.paths import TOOLS_ROOT


def _source_environment() -> dict[str, str]:
    inherited = os.environ.get("PYTHONPATH", "")
    return {
        **os.environ,
        "PYTHONPATH": os.pathsep.join(
            value for value in (str(TOOLS_ROOT / "src"), inherited) if value
        ),
    }


def test_importing_numsim_checkers_does_not_import_other_simulators():
    script = """
import json
import sys
import tirx_harness.numsim
import tirx_harness.numsim.checkers
print(json.dumps(sorted(sys.modules)))
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=TOOLS_ROOT,
        env=_source_environment(),
        check=True,
        capture_output=True,
        text=True,
    )

    imported = set(json.loads(completed.stdout))
    assert not any("nymph" in name.lower() for name in imported)


def test_root_checker_facades_are_lazy():
    script = """
import json
import sys
import tirx_harness
assert callable(tirx_harness.synccheck)
assert callable(tirx_harness.racecheck)
print(json.dumps(sorted(sys.modules)))
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=TOOLS_ROOT,
        env=_source_environment(),
        check=True,
        capture_output=True,
        text=True,
    )

    imported = set(json.loads(completed.stdout))
    assert not any(name.startswith("tirx_harness.numsim") for name in imported)
