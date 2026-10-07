"""Serializable TIRx source metadata."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


def _required_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_name} must be an integer")
    return value


@dataclass(frozen=True)
class SourceSpan:
    """One stable, serializable source range."""

    source_name: str
    line: int
    column: int
    end_line: int
    end_column: int

    def __post_init__(self) -> None:
        if not isinstance(self.source_name, str) or not self.source_name:
            raise ValueError("source span source_name must be a non-empty string")
        coordinates = {
            "line": self.line,
            "column": self.column,
            "end_line": self.end_line,
            "end_column": self.end_column,
        }
        for name, value in coordinates.items():
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"source span {name} must be a positive integer")
        if self.end_line < self.line or (
            self.end_line == self.line and self.end_column < self.column
        ):
            raise ValueError("source span end must not precede its start")

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "span",
            "source_name": self.source_name,
            "line": self.line,
            "column": self.column,
            "end_line": self.end_line,
            "end_column": self.end_column,
        }

    @classmethod
    def from_dict(cls, value: Any) -> SourceSpan:
        if not isinstance(value, dict) or value.get("kind") != "span":
            raise ValueError("source span must be a kind='span' dictionary")
        expected = {"kind", "source_name", "line", "column", "end_line", "end_column"}
        if set(value) != expected:
            raise ValueError("source span contains missing or unexpected fields")
        source_name = value.get("source_name")
        if not isinstance(source_name, str):
            raise ValueError("source span source_name must be a string")
        return cls(
            source_name=source_name,
            line=_required_int(value.get("line"), "source span line"),
            column=_required_int(value.get("column"), "source span column"),
            end_line=_required_int(value.get("end_line"), "source span end_line"),
            end_column=_required_int(value.get("end_column"), "source span end_column"),
        )


@dataclass(frozen=True)
class SequentialSourceSpan:
    """Ordered provenance for an inline/macro-expanded IR operation."""

    spans: tuple[SerializedSourceSpan, ...]

    def __post_init__(self) -> None:
        if not self.spans:
            raise ValueError("sequential source span must contain at least one span")
        if not all(isinstance(span, (SourceSpan, SequentialSourceSpan)) for span in self.spans):
            raise ValueError("sequential source span contains an invalid child")

    def to_dict(self) -> dict[str, Any]:
        return {"kind": "sequential", "spans": [span.to_dict() for span in self.spans]}

    @classmethod
    def from_dict(cls, value: Any) -> SequentialSourceSpan:
        if not isinstance(value, dict) or value.get("kind") != "sequential":
            raise ValueError("sequential source span must be a kind='sequential' dictionary")
        if set(value) != {"kind", "spans"}:
            raise ValueError("sequential source span contains missing or unexpected fields")
        spans = value.get("spans")
        if not isinstance(spans, list):
            raise ValueError("sequential source span spans must be a list")
        return cls(tuple(deserialize_source_span(span, allow_none=False) for span in spans))


SerializedSourceSpan = SourceSpan | SequentialSourceSpan


def deserialize_source_span(value: Any, *, allow_none: bool = True) -> SerializedSourceSpan | None:
    """Restore one canonical structured source span from JSON data."""

    if value is None and allow_none:
        return None
    if not isinstance(value, dict):
        raise ValueError("source span must be a dictionary or null")
    kind = value.get("kind")
    if kind == "span":
        return SourceSpan.from_dict(value)
    if kind == "sequential":
        return SequentialSourceSpan.from_dict(value)
    raise ValueError(f"source span has unsupported kind {kind!r}")


def serialize_source_span(value: SerializedSourceSpan | None) -> dict[str, Any] | None:
    return None if value is None else value.to_dict()


def source_span_from_tvm(value: Any) -> SerializedSourceSpan | None:
    """Convert a TVM Span/SequentialSpan without using its unstable string form."""

    if value is None:
        return None
    children = getattr(value, "spans", None)
    if children is not None:
        spans = tuple(
            child_span
            for child in children
            if (child_span := source_span_from_tvm(child)) is not None
        )
        return None if not spans else SequentialSourceSpan(spans)
    try:
        source_name = str(value.source_name.name)
        return SourceSpan(
            source_name=source_name,
            line=int(value.line),
            column=int(value.column),
            end_line=int(value.end_line),
            end_column=int(value.end_column),
        )
    except (AttributeError, TypeError, ValueError):
        # Hand-built IR and older TIRx versions may not expose a usable span.
        return None


def source_span_from_node(node: Any) -> SerializedSourceSpan | None:
    return source_span_from_tvm(getattr(node, "span", None))


def flatten_source_span(value: SerializedSourceSpan | None) -> tuple[SourceSpan, ...]:
    if value is None:
        return ()
    if isinstance(value, SourceSpan):
        return (value,)
    return tuple(leaf for span in value.spans for leaf in flatten_source_span(span))


def format_source_span(value: SerializedSourceSpan | None) -> str | None:
    """Return a compact human-readable location without losing inline context."""

    spans = flatten_source_span(value)
    if not spans:
        return None

    def location(span: SourceSpan) -> str:
        return f"{span.source_name}:{span.line}:{span.column}"

    primary = location(spans[0])
    if len(spans) == 1:
        return primary
    return f"{primary} (expanded through {' -> '.join(location(span) for span in spans[1:])})"


@dataclass(frozen=True)
class SourceEntry:
    """One IR occurrence and its bounded diagnostic text.

    Sequences report their length; loops, branches, and attributes elide bodies;
    those child occurrences retain their own entries and source spans.
    """

    op_id: int
    kind: str
    text: str
    span: SerializedSourceSpan | None = None
    op_name: str | None = None
    node: Any | None = field(default=None, repr=False, compare=False, hash=False)
    resolved: Any | None = field(default=None, repr=False, compare=False, hash=False)

    def to_dict(self, *, include_span: bool = True) -> dict:
        result = {
            "op_id": self.op_id,
            "kind": self.kind,
            "text": self.text,
            "span": serialize_source_span(self.span) if include_span else None,
        }
        if self.op_name is not None:
            result["op_name"] = self.op_name
        return result


__all__ = [
    "SequentialSourceSpan",
    "SerializedSourceSpan",
    "SourceEntry",
    "SourceSpan",
    "deserialize_source_span",
    "flatten_source_span",
    "format_source_span",
    "serialize_source_span",
    "source_span_from_node",
    "source_span_from_tvm",
]
