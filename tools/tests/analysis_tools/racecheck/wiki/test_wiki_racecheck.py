"""Complete native Racecheck gate for the established wiki-kernel families."""

from __future__ import annotations

from collections import Counter

import pytest

from tirx_harness import racecheck

from ._cases import WIKI_RACECHECK_SPECS, prepare_wiki_racecheck_case


def _matches_expected_review(finding, spec) -> bool:
    expectation = spec.expected_uninitialized_read
    if expectation is None or finding.kind != "uninitialized_read":
        return False
    details = finding.details
    source_anchor = details.get("source_anchor")
    source_span = source_anchor.get("source_span") if isinstance(source_anchor, dict) else None
    if source_anchor is None or not isinstance(source_anchor, dict):
        return False
    if source_anchor.get("source_span") is None:
        return (
            expectation.allow_spanless_source
            and details.get("space") == expectation.space
            and details.get("byte_len") == 4
            and bool(source_anchor.get("source_text", "").strip())
        )
    return (
        details.get("space") == expectation.space
        and details.get("byte_len") == 4
        and bool(source_anchor.get("source_text", "").strip())
        and isinstance(source_span, dict)
        and source_span.get("source_name", "").endswith(expectation.source_suffix)
    )


def _is_auditable_review(finding, spec) -> bool:
    if finding.status != "review":
        return False
    if _matches_expected_review(finding, spec):
        return True
    if finding.kind == "alias_stale_read":
        return True
    if finding.details.get("access_pair") not in {"read_write", "write_read", "write_write"}:
        return False

    endpoints = (finding.details.get("prior"), finding.details.get("current"))
    if not all(isinstance(endpoint, dict) for endpoint in endpoints):
        return False
    if any(endpoint.get("space") != "tmem" for endpoint in endpoints):
        return False
    operations = tuple(endpoint.get("operation") for endpoint in endpoints)
    if not all(isinstance(operation, dict) for operation in operations):
        return False
    if tuple(endpoint.get("access_kind") for endpoint in endpoints) != ("read", "write"):
        return False
    if finding.details.get("ordering_failure") != "async_lifetime_not_drained":
        return False
    return all(
        isinstance(operation.get("source"), dict)
        and bool(operation["source"].get("source_text", "").strip())
        for operation in operations
    )


@pytest.mark.parametrize("spec", WIKI_RACECHECK_SPECS, ids=lambda spec: spec.case_id)
def test_wiki_kernel_matches_racecheck_contract(spec) -> None:
    case = prepare_wiki_racecheck_case(spec)
    report = racecheck(case.kernel, case.args)

    assert report.verdict in {"clean", "review"}, report.format()
    assert report.to_dict()["native"]["incomplete"] == [], report.format()
    assert all(_is_auditable_review(finding, spec) for finding in report.findings), report.format()
    if spec.expected_uninitialized_read is not None:
        assert Counter(finding.kind for finding in report.findings) == {
            "uninitialized_read": spec.expected_uninitialized_read.count
        }, report.format()
