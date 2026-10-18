"""Two-shot parity cannot use missing, unrelated, incorrect, or slow evidence."""

from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from benchmarks.multimem import common
from benchmarks.multimem.two_shot_all_reduce import validate_rows


@pytest.mark.parametrize("delta", (
    {"comparison_kind": "historical"}, {"bitwise_equal": False},
    {"ratio": 1.02}, {"ratio": float("nan")}, {"ratio": 0},
))
def test_invalid_evidence_is_rejected(delta):
    row = {"M": 1024, "N": 512, "ratio": 1.0, "bitwise_equal": True,
           "comparison_kind": "port_fidelity", **delta}
    with pytest.raises(RuntimeError, match="Two-shot fidelity validation failed"):
        validate_rows([row], [[1024, 512]])


def test_all_requested_shapes_must_pass():
    row = {"M": 1024, "N": 512, "ratio": 1.0, "bitwise_equal": True,
           "comparison_kind": "port_fidelity"}
    validate_rows([row], [[1024, 512]])
    with pytest.raises(RuntimeError, match="2048x512"):
        validate_rows([row], [[1024, 512], [2048, 512]])


@pytest.mark.parametrize("arguments,world,tiles,seed", [
    ([], 4, None, 0),
    (["--world", "2", "--tiles", "4", "--seed", "7"], 2, 4, 7),
])
def test_common_preflight_cli(arguments, world, tiles, seed, monkeypatch, capsys):
    run = Mock(return_value=SimpleNamespace(outputs={"data_out": None, "flag": None}))
    monkeypatch.setattr(common, "run_preflight", run)
    monkeypatch.setattr(sys, "argv", ["benchmarks.multimem.common", *arguments])

    common.main()

    run.assert_called_once_with(world, tiles=tiles, seed=seed)
    output = capsys.readouterr().out
    assert f"NumSim/Racecheck/Synccheck passed for {world} ranks" in output
    assert "outputs: ['data_out', 'flag']" in output


def test_common_preflight_help_does_not_import_kernel_dependencies():
    result = subprocess.run(
        [sys.executable, "-c", "\n".join([
            "import runpy, sys",
            "sys.modules.update(dict.fromkeys(('tvm', 'torch', 'tirx_harness', 'ported')))",
            "sys.argv = ['benchmarks.multimem.common', '--help']",
            "runpy.run_module('benchmarks.multimem.common', run_name='__main__')",
        ])],
        cwd=Path(common.__file__).parents[2],
        capture_output=True,
        text=True,
        check=True,
    )

    assert "CPU preflight" in result.stdout
    assert all(option in result.stdout for option in ("--world", "--tiles", "--seed"))
