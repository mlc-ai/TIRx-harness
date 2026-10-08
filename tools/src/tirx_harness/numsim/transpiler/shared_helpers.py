"""Static TIR-inspection and text helpers shared by the host-side modules."""

from __future__ import annotations

from typing import Any


def plain_text(value: Any) -> str:
    """Render ``value`` as an ordinary ``str``.

    TVM FFI returns ``String`` subclasses of ``str``; copying through UTF-8
    detaches the result from the FFI object so it can be cached, hashed, and
    embedded in generated source without keeping the TIR node alive.
    """

    text = str(value)
    return text.encode("utf-8").decode("utf-8")
