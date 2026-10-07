"""Shared verdict model for TIRx analysis tools."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, fields, is_dataclass
from typing import Any

_VERDICT_PRIORITY = {"clean": 0, "review": 1, "incomplete": 2, "error": 3}
_FINDING_STATUSES = frozenset(_VERDICT_PRIORITY) - {"clean"}
_FINDING_SORT_PRIORITY = {"error": 0, "incomplete": 1, "review": 2}


def _json_safe(value: Any):
    if is_dataclass(value):
        return {item.name: _json_safe(getattr(value, item.name)) for item in fields(value)}
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (set, frozenset)):
        items = [_json_safe(v) for v in value]
        return sorted(items, key=lambda item: json.dumps(item, sort_keys=True, default=str))
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def stable_finding_id(checker: str, kind: str, *identity: Any) -> str:
    """Build a deterministic compact ID from caller-supplied semantic identity."""

    payload = json.dumps(_json_safe(identity), sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(payload.encode()).hexdigest()[:16]
    return f"{checker}:{kind}:{digest}"


@dataclass(frozen=True)
class Finding:
    """One actionable checker result.

    ``error`` means a confirmed kernel/protocol defect. ``incomplete`` records
    the specific missing evidence or coverage that prevented certification.
    ``review`` records an advisory or potential defect that requires
    investigation; when it was produced from a provisional trace, a separate
    ``incomplete`` finding names the blocker.
    """

    id: str
    status: str
    kind: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.status not in _FINDING_STATUSES:
            raise ValueError(f"invalid finding status: {self.status!r}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "status": self.status,
            "kind": self.kind,
            "message": self.message,
            "details": _json_safe(self.details),
        }


class CheckFailed(RuntimeError):
    """Raised when ``require_clean()`` receives a non-clean report."""


def sort_findings(findings) -> list[Finding]:
    """Return findings in the shared render/gate order."""

    return sorted(
        findings,
        key=lambda finding: (_FINDING_SORT_PRIORITY[finding.status], finding.kind, finding.id),
    )


class VerdictMixin:
    """Minimal common report surface for strict analysis reports."""

    checker_name = "checker"

    def _collect_findings(self) -> list[Finding]:
        raise NotImplementedError

    @property
    def findings(self) -> list[Finding]:
        return sort_findings(self._collect_findings())

    @property
    def verdict(self) -> str:
        findings = self.findings
        if not findings:
            return "clean"
        return max(findings, key=lambda finding: _VERDICT_PRIORITY[finding.status]).status

    def require_clean(self) -> None:
        verdict = self.verdict
        if verdict == "clean":
            return
        counts = {status: 0 for status in _FINDING_STATUSES}
        for finding in self.findings:
            counts[finding.status] += 1
        summary = ", ".join(f"{status}={counts[status]}" for status in sorted(counts))
        formatter = getattr(self, "format", None)
        report = formatter() if callable(formatter) else repr(self)
        raise CheckFailed(f"{self.checker_name} {verdict} ({summary})\n{report}")

    def _finding_lines(self, findings: list[Finding] | None = None) -> list[str]:
        """Render every typed finding so ``print()`` never hides a verdict cause."""

        findings = self.findings if findings is None else findings
        if not findings:
            return []
        lines = [f"=== Verdict findings ({len(findings)}) ==="]
        for finding in findings:
            message_lines = str(finding.message).splitlines() or [""]
            lines.append(f"  [{finding.status.upper()}] {finding.kind}: {message_lines[0]}")
            lines.extend(f"    {line}" for line in message_lines[1:])
        lines.append("")
        return lines
