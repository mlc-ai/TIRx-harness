"""Strict report adapters for the native Synccheck and Racecheck engines."""

from __future__ import annotations

import json
from typing import Any, ClassVar

from tirx_harness._report import Finding, VerdictMixin, stable_finding_id
from .checker_render import (
    _native_execution_label,
    _native_finding_evidence_lines,
    _native_finding_headline,
    _native_finding_identity,
)


class NativeCheckerReport(VerdictMixin):
    """Common strict surface for one native analysis payload."""

    checker_name: ClassVar[str] = "checker"
    valid_verdicts: ClassVar[frozenset[str]] = frozenset()

    def __init__(self, native_payload: dict[str, Any]):
        if not isinstance(native_payload, dict):
            raise TypeError(f"native {self.checker_name} payload must be a dict")
        verdict = native_payload.get("verdict")
        if verdict not in self.valid_verdicts:
            raise ValueError(f"native {self.checker_name} payload has invalid verdict {verdict!r}")
        self.native_payload = native_payload
        self._findings_cache: tuple[Finding, ...] | None = None

    @classmethod
    def from_native(cls, payload: dict[str, Any]):
        return cls(payload)

    @property
    def verdict(self) -> str:
        return str(self.native_payload["verdict"])

    def _finding(
        self,
        status: str,
        kind: str,
        message: str,
        details: dict[str, Any],
    ) -> Finding:
        return Finding(
            id=stable_finding_id(
                self.checker_name,
                kind,
                _native_finding_identity(details),
            ),
            status=status,
            kind=kind,
            message=message,
            details=details,
        )

    def _finding_lines(self, findings: list[Finding]) -> list[str]:
        if not findings:
            return []
        lines = [f"=== Verdict findings ({len(findings)}) ==="]
        for finding in findings:
            headline = _native_finding_headline(finding)
            lines.append(f"  [{finding.status.upper()}] {finding.kind}: {headline}")
            message_lines = str(finding.message).splitlines() or [""]
            if headline == message_lines[0]:
                lines.extend(f"    {line}" for line in message_lines[1:])
            for key, label in (("access_pair", "Access pair"), ("reason", "Reason"), ("cause", "Cause")):
                if key in finding.details:
                    lines.append(f"    {label}: {finding.details[key]}")
            lines.extend(_native_finding_evidence_lines(finding))
            hint = finding.details.get("hint")
            if hint:
                lines.append(f"    Hint: {hint}")
        lines.append("")
        return lines

    def to_dict(self) -> dict[str, Any]:
        findings = self.findings
        return {
            "schema_version": 4,
            "checker": self.checker_name,
            "engine": "native",
            "verdict": self.verdict,
            "findings": [finding.to_dict() for finding in findings],
            "summary": (
                f"{self.checker_name} {self.verdict.upper()} "
                f"({len(findings)} findings, native execution)"
            ),
            "native": self.native_payload,
        }

    def print(self) -> None:
        print(self.format())


class SyncCheckReport(NativeCheckerReport):
    """Public report for one native Synccheck execution."""

    checker_name = "synccheck"
    valid_verdicts = frozenset({"clean", "review", "incomplete", "error"})

    def _collect_findings(self) -> list[Finding]:
        if self._findings_cache is not None:
            return list(self._findings_cache)

        payload = self.native_payload
        findings: dict[str, Finding] = {}

        def add(status: str, kind: str, message: str, details: dict[str, Any]) -> None:
            finding = self._finding(status, kind, message, details)
            findings.setdefault(finding.id, finding)

        for raw in payload.get("findings", ()):
            details = raw if isinstance(raw, dict) else {"finding": raw}
            kind = str(details.get("kind", "protocol_error"))
            status = str(details.get("status", "error"))
            if status not in {"review", "error"}:
                status = "error"
            add(status, kind, str(details.get("message", kind.replace("_", " "))), details)

        for raw in payload.get("advisories", ()):
            details = raw if isinstance(raw, dict) else {"advisory": raw}
            kind = str(details.get("kind", "native_advisory"))
            add("review", kind, str(details.get("message", kind.replace("_", " "))), details)

        for raw in payload.get("incomplete", ()):
            details = raw if isinstance(raw, dict) else {"reason": raw}
            kind = str(details.get("kind", "analysis_incomplete"))
            add(
                "incomplete",
                kind,
                str(details.get("message", kind.replace("_", " "))),
                details,
            )

        for raw in payload.get("advisories", ()):
            details = raw if isinstance(raw, dict) else {"advisory": raw}
            kind = str(details.get("kind", "native_advisory"))
            add("review", kind, str(details.get("message", kind.replace("_", " "))), details)

        execution_error = payload.get("execution_error")
        if isinstance(execution_error, dict):
            kind = str(execution_error.get("kind", "execution_error"))
            has_error = any(finding.status == "error" for finding in findings.values())
            if not has_error and kind not in {
                "poll_limit",
                "transition_limit",
                "choice_prefix_replay_divergence",
                "analysis_incomplete",
            }:
                add(
                    "error",
                    kind,
                    str(execution_error.get("message", kind.replace("_", " "))),
                    execution_error,
                )

        statuses = {finding.status for finding in findings.values()}
        source_anchor = payload.get("source_anchor")
        source_details = {"source_anchor": source_anchor} if isinstance(source_anchor, dict) else {}
        if self.verdict == "error" and "error" not in statuses:
            add(
                "error",
                "native_execution_error",
                "native synccheck reported an error without a typed finding",
                {"payload_verdict": self.verdict, **source_details},
            )
        if self.verdict == "incomplete" and "incomplete" not in statuses:
            add(
                "incomplete",
                "analysis_incomplete",
                "native synccheck did not exhaust the requested analysis coverage",
                {"payload_verdict": self.verdict, **source_details},
            )
        if self.verdict == "review" and "review" not in statuses:
            add(
                "review",
                "native_advisory_untyped",
                "native synccheck requested review without a typed advisory",
                {"payload_verdict": self.verdict, **source_details},
            )

        self._findings_cache = tuple(findings.values())
        return list(self._findings_cache)

    def to_dict(self) -> dict[str, Any]:
        result = super().to_dict()
        result["strategies"] = ["native"]
        return result

    def format(self) -> str:
        findings = self.findings
        phase = self.native_payload.get("phase", {})
        phase_label = phase.get("name", phase.get("index", "?"))
        execution_label = _native_execution_label(self.native_payload)
        lines = [
            f"synccheck {self.verdict.upper()} - {len(findings)} finding(s); "
            f"native phase {phase_label}, {execution_label}"
        ]
        finding_lines = self._finding_lines(findings)
        if finding_lines:
            lines.extend(["", *finding_lines])
        return "\n".join(lines).rstrip()


class RaceReport(NativeCheckerReport):
    """Public report for one native Racecheck execution."""

    checker_name = "racecheck"
    valid_verdicts = frozenset({"clean", "review", "incomplete", "error"})

    def _collect_findings(self) -> list[Finding]:
        if self._findings_cache is not None:
            return list(self._findings_cache)

        payload = self.native_payload
        findings: dict[str, Finding] = {}

        def add(status: str, kind: str, message: str, details: dict[str, Any]) -> None:
            finding = self._finding(status, kind, message, details)
            findings.setdefault(finding.id, finding)

        for raw in payload.get("findings", ()):
            details = raw if isinstance(raw, dict) else {"finding": raw}
            kind = str(details.get("kind", "physical_race"))
            status = str(details.get("status", "error"))
            if status not in {"review", "error"}:
                status = "error"
            add(status, kind, str(details.get("message", kind.replace("_", " "))), details)

        for raw in payload.get("advisories", ()):
            details = raw if isinstance(raw, dict) else {"advisory": raw}
            kind = str(details.get("kind", "native_advisory"))
            add("review", kind, str(details.get("message", kind.replace("_", " "))), details)

        def incomplete_key(raw: Any) -> str:
            details = raw if isinstance(raw, dict) else {"reason": raw}
            semantic = {
                key: value
                for key, value in details.items()
                if key not in {"static_site", "source_anchor", "occurrence_count", "cluster_ids"}
            }
            operation = semantic.get("operation")
            if isinstance(operation, dict):
                # Source-map enrichment does not create a second occurrence.
                semantic["operation"] = {
                    key: value for key, value in operation.items() if key != "source"
                }
            return json.dumps(semantic, sort_keys=True, separators=(",", ":"), default=str)

        top_incomplete = {incomplete_key(raw) for raw in payload.get("incomplete", ())}
        sync = payload.get("sync")
        if isinstance(sync, dict):
            for raw in sync.get("findings", ()):
                raw_details = raw if isinstance(raw, dict) else {"finding": raw}
                details = {**raw_details, "domain": "synchronization"}
                kind = str(details.get("kind", "sync_protocol_error"))
                add("error", kind, str(details.get("message", kind.replace("_", " "))), details)
            for raw in sync.get("incomplete", ()):
                raw_details = raw if isinstance(raw, dict) else {"reason": raw}
                if incomplete_key(raw_details) in top_incomplete:
                    continue
                details = {**raw_details, "domain": "synchronization"}
                kind = str(details.get("kind", "analysis_incomplete"))
                add(
                    "incomplete",
                    kind,
                    str(details.get("message", kind.replace("_", " "))),
                    details,
                )

        for raw in payload.get("incomplete", ()):
            details = raw if isinstance(raw, dict) else {"reason": raw}
            kind = str(details.get("kind", "analysis_incomplete"))
            add(
                "incomplete",
                kind,
                str(details.get("message", kind.replace("_", " "))),
                details,
            )

        execution_error = payload.get("execution_error")
        has_error = any(finding.status == "error" for finding in findings.values())
        if isinstance(execution_error, dict) and not has_error:
            kind = str(execution_error.get("kind", "execution_error"))
            if kind not in {"poll_limit", "transition_limit", "analysis_incomplete"}:
                add(
                    "error",
                    kind,
                    str(execution_error.get("message", kind.replace("_", " "))),
                    execution_error,
                )

        statuses = {finding.status for finding in findings.values()}
        fallback = {
            "error": (
                "native_execution_error",
                "native racecheck reported an error without a typed finding",
            ),
            "incomplete": (
                "analysis_incomplete",
                "native racecheck did not exhaust the requested analysis coverage",
            ),
            "review": (
                "native_advisory_untyped",
                "native racecheck requested review without a typed advisory",
            ),
        }
        if self.verdict in fallback and self.verdict not in statuses:
            kind, message = fallback[self.verdict]
            add(self.verdict, kind, message, {"payload_verdict": self.verdict})

        self._findings_cache = tuple(findings.values())
        return list(self._findings_cache)

    def format(self) -> str:
        findings = self.findings
        lines = [f"=== racecheck {self.verdict.upper()} ==="]
        lines.extend(self._finding_lines(findings))
        access_count = self.native_payload.get("access_count", 0)
        phase = self.native_payload.get("phase", {})
        phase_label = phase.get("name", phase.get("index", "?"))
        checked_spaces = self.native_payload.get("checked_memory_spaces", [])
        if checked_spaces:
            scope = f"race conflicts checked in {', '.join(map(str, checked_spaces))}"
            if "global" not in checked_spaces:
                scope += "; global accesses execute but are not race-classified"
            lines.append(scope)
        lines.append(f"native phase {phase_label}: {access_count} semantic memory access(es)")
        return "\n".join(lines).rstrip()


__all__ = ["NativeCheckerReport", "RaceReport", "SyncCheckReport"]
