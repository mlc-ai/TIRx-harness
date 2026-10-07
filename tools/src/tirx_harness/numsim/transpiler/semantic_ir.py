"""Span-insensitive serialization for semantic TIRx cache identities."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

from . import native_frontend


_SERIALIZATIONS: ContextVar[dict[int, tuple[Any, str]] | None] = ContextVar(
    "numsim_semantic_ir_serializations", default=None
)


@contextmanager
def cache_semantic_ir_serializations() -> Iterator[None]:
    """Reuse immutable IR serialization only for the lifetime of one transpile."""
    token = _SERIALIZATIONS.set({})
    try:
        yield
    finally:
        _SERIALIZATIONS.reset(token)


def semantic_ir_json(value: Any) -> str:
    """Serialize IR after disconnecting diagnostic-only source spans.

    TVM's JSON graph includes source paths and coordinates even though spans do
    not participate in structural equality.  Clearing every reflected ``span``
    edge and round-tripping the graph drops the now-unreachable Span and
    SourceName nodes, leaving a deterministic semantic serialization.
    """

    serializations = _SERIALIZATIONS.get()
    if serializations is not None:
        cached = serializations.get(id(value))
        if cached is not None and cached[0] is value:
            return cached[1]

    serialized = native_frontend.semantic_ir_json(value)
    if serializations is not None:
        serializations[id(value)] = (value, serialized)
    return serialized


__all__ = ["cache_semantic_ir_serializations", "semantic_ir_json"]
