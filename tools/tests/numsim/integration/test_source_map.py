from __future__ import annotations

import json

import pytest
from tvm.ir import SequentialSpan, SourceName, Span

from tirx_harness.numsim.transpiler.source_map import (
    SequentialSourceSpan,
    SourceEntry,
    SourceSpan,
    deserialize_source_span,
    flatten_source_span,
    source_span_from_tvm,
)


class RepeatedNode:
    span = None

    def __init__(self, identity: object) -> None:
        self.identity = identity

    def __str__(self) -> str:
        return "repeated-node"

    def same_as(self, other: object) -> bool:
        return isinstance(other, RepeatedNode) and self.identity is other.identity


def _entry(op_id: int, node: RepeatedNode) -> SourceEntry:
    return SourceEntry(
        op_id=op_id,
        kind="RepeatedNode",
        text="repeated-node",
        node=node,
    )


def test_source_entry_manifest_excludes_process_local_node_identity() -> None:
    node = RepeatedNode(object())

    assert _entry(5, node).to_dict() == {
        "op_id": 5,
        "kind": "RepeatedNode",
        "text": "repeated-node",
        "span": None,
    }


def test_regular_span_uses_stable_structured_serialization() -> None:
    raw = Span(SourceName("/repo/kernels/attention.py"), 142, 142, 17, 40)

    span = source_span_from_tvm(raw)

    assert span == SourceSpan("/repo/kernels/attention.py", 142, 17, 142, 40)
    entry = SourceEntry(5, "Call", "T.cuda.mbarrier_wait(...)", span=span)
    payload = entry.to_dict()["span"]
    assert payload == {
        "kind": "span",
        "source_name": "/repo/kernels/attention.py",
        "line": 142,
        "column": 17,
        "end_line": 142,
        "end_column": 40,
    }
    assert "0x" not in json.dumps(payload)
    assert deserialize_source_span(payload) == span


def test_sequential_span_preserves_and_flattens_inline_provenance_order() -> None:
    call_site = Span(SourceName("/repo/kernels/attention.py"), 142, 142, 17, 40)
    wrapper = Span(SourceName("/repo/lang/pipeline.py"), 110, 110, 9, 28)
    primitive = Span(SourceName("/repo/lang/mbarrier.py"), 44, 44, 5, 63)
    raw = SequentialSpan([call_site, SequentialSpan([wrapper, primitive])])

    span = source_span_from_tvm(raw)

    assert isinstance(span, SequentialSourceSpan)
    assert [leaf.source_name for leaf in flatten_source_span(span)] == [
        "/repo/kernels/attention.py",
        "/repo/lang/pipeline.py",
        "/repo/lang/mbarrier.py",
    ]
    payload = span.to_dict()
    assert payload["kind"] == "sequential"
    assert [child["kind"] for child in payload["spans"]] == ["span", "span", "span"]
    assert deserialize_source_span(payload) == span


def test_stringified_span_is_not_accepted_as_structured_metadata() -> None:
    with pytest.raises(ValueError, match="dictionary or null"):
        deserialize_source_span("Span(SourceName(kernel.py, 0x1234), 1, 1, 1, 2)")
