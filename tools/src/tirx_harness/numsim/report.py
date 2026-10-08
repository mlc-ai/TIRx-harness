"""Structured NumSim comparison reports."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Mismatch:
    output: str
    index: tuple[int, ...]
    actual: object
    expected: object

    def render(self) -> str:
        return (
            f"output={self.output!r} index={self.index} actual={self.actual!r} "
            f"expected={self.expected!r}"
        )


@dataclass
class NumSimReport:
    ok: bool
    mismatches: list[Mismatch] = field(default_factory=list)
    diagnostics: list[dict] = field(default_factory=list)

    @property
    def verdict(self) -> str:
        if not self.ok:
            return "error"
        if any(item.get("status") == "review" for item in self.diagnostics):
            return "review"
        return "clean"

    def require_ok(self) -> None:
        if not self.ok:
            first = self.mismatches[0] if self.mismatches else None
            detail = "<missing>" if first is None else first.render()
            raise AssertionError(f"NumSim comparison failed; first mismatch: {detail}")
