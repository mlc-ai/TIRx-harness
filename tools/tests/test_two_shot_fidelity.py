"""Two-shot parity cannot use missing, unrelated, incorrect, or slow evidence."""

import pytest

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
