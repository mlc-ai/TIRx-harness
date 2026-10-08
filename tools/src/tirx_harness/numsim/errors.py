"""NumSim exception hierarchy."""

from __future__ import annotations

from typing import Any


class NumSimError(RuntimeError):
    """Base class for NumSim failures."""


class UnsupportedTIRxError(NumSimError):
    """Raised when fail-closed transpilation finds unsupported TIRx behavior."""

    def __init__(
        self,
        message: str,
        *,
        unsupported: tuple[str, ...] = (),
        source_span: Any = None,
    ) -> None:
        super().__init__(message)
        self.unsupported = unsupported
        # Span of the node that made the kernel unsupported, when the raiser
        # knows it. Reporting uses this directly, so no whole-function anchor
        # has to be reconstructed for a finding that is really about one node.
        self.source_span = source_span


class UnmodeledTIRxFormError(UnsupportedTIRxError):
    """Raised for a valid public TIRx form NumSim intentionally does not model."""

    def __init__(self, target_id: str, message: str) -> None:
        if not target_id.startswith(("call:", "tile:")):
            raise ValueError(f"invalid NumSim form target {target_id!r}")
        super().__init__(message)
        self.target_id = target_id


class NumSimBuildError(NumSimError):
    """Raised when generated Rust cannot be built or loaded."""


class NumSimExecutionError(NumSimError):
    """Raised when a compiled NumSim artifact fails during execution."""
