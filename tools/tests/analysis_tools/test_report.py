import pytest

from tirx_harness._report import Finding, VerdictMixin, sort_findings, stable_finding_id


class _Report(VerdictMixin):
    checker_name = "testcheck"

    def __init__(self, findings=()):
        self._findings = list(findings)

    def _collect_findings(self):
        return list(self._findings)

    def format(self):
        return "formatted report"


def _finding(status: str, kind: str = "test") -> Finding:
    return Finding(
        id=stable_finding_id("testcheck", kind, "stable identity"),
        status=status,
        kind=kind,
        message=f"{status} finding",
    )


@pytest.mark.parametrize(
    ("statuses", "expected"),
    [
        ([], "clean"),
        (["review"], "review"),
        (["review", "incomplete"], "incomplete"),
        (["review", "incomplete", "error"], "error"),
    ],
)
def test_verdict_uses_strict_precedence(statuses, expected):
    report = _Report([_finding(status, f"kind_{idx}") for idx, status in enumerate(statuses)])
    assert report.verdict == expected


def test_require_clean_is_the_single_strict_gate():
    _Report().require_clean()

    for status in ("review", "incomplete", "error"):
        with pytest.raises(RuntimeError, match=rf"testcheck {status}"):
            _Report([_finding(status)]).require_clean()


def test_finding_id_is_stable_and_payload_is_json_safe():
    first = stable_finding_id("racecheck", "symbolic_access", "smem", (0, 1))
    second = stable_finding_id("racecheck", "symbolic_access", "smem", (0, 1))
    assert first == second

    finding = Finding(
        id=first,
        status="incomplete",
        kind="symbolic_access",
        message="access could not be resolved",
        details={"region": (0, 1)},
    )
    assert finding.to_dict()["details"]["region"] == [0, 1]


def test_finding_rejects_non_actionable_status():
    with pytest.raises(ValueError, match="invalid finding status"):
        _finding("clean")


def test_sort_findings_uses_status_then_kind_then_id():
    findings = [
        Finding("z", "review", "a", "review"),
        Finding("b", "incomplete", "z", "incomplete z"),
        Finding("a", "incomplete", "z", "incomplete z first id"),
        Finding("c", "error", "m", "error"),
        Finding("d", "incomplete", "a", "incomplete a"),
    ]

    assert [(finding.status, finding.kind, finding.id) for finding in sort_findings(findings)] == [
        ("error", "m", "c"),
        ("incomplete", "a", "d"),
        ("incomplete", "z", "a"),
        ("incomplete", "z", "b"),
        ("review", "a", "z"),
    ]


def test_signal_protocol_report_preserves_roles_message_and_hint():
    from tirx_harness.numsim.checker_report import RaceReport

    def operation(warp, site):
        return {
            "kernel_index": 0,
            "global_warp_id": warp,
            "per_warp_sequence": site,
            "source_op_id": site,
            "loop_frames": [],
        }

    report = RaceReport(
        {
            "verdict": "error",
            "findings": [
                {
                    "status": "error",
                    "kind": "signal_protocol_error",
                    "message": "Signal access has no happens-before relationship with wait_until.",
                    "hint": "Order plain initialization/reset with synchronization.",
                    "wait_operation": operation(0, 10),
                    "plain_operation": operation(1, 20),
                }
            ],
        }
    )
    text = report.format()
    assert "Signal access has no happens-before relationship" in text
    assert "Wait: warp 0" in text
    assert "Plain access: warp 1" in text
    assert "Hint: Order plain initialization/reset" in text
    assert "Related:" not in text
    assert report.to_dict()["findings"][0]["details"]["hint"].startswith("Order plain")


@pytest.mark.parametrize(
    "reason",
    [
        "wait_exit_unproven",
        "signal_history_truncated",
        "signal_write_not_recorded",
    ],
)
def test_signal_coverage_gap_is_analysis_incomplete_not_a_kernel_error(reason):
    from tirx_harness.numsim.checker_report import RaceReport

    report = RaceReport(
        {
            "verdict": "incomplete",
            "incomplete": [
                {
                    "kind": "analysis_incomplete",
                    "reason": reason,
                    "message": "Cannot finish signal analysis with the available evidence.",
                    "hint": "Retain a reproducer of the missing evidence.",
                }
            ],
        }
    )
    assert [(f.status, f.kind) for f in report.findings] == [
        ("incomplete", "analysis_incomplete"),
    ]
    assert report.findings[0].details["reason"] == reason
    assert f"Reason: {reason}" in report.format()
    assert "Cannot finish signal analysis" in report.format()
    assert "Hint: Retain a reproducer" in report.format()


@pytest.mark.parametrize("checker", ["racecheck", "synccheck"])
def test_incomplete_reasons_keep_distinct_identities_and_common_kind(checker):
    from tirx_harness.numsim.checker_report import RaceReport, SyncCheckReport

    report_cls = RaceReport if checker == "racecheck" else SyncCheckReport
    reasons = ["resource_limit", "signal_history_truncated", "native_frontend_unsupported"]
    records = [{"kind": "analysis_incomplete", "reason": reason} for reason in reasons]
    payload = {"verdict": "incomplete", "incomplete": records}
    if checker == "racecheck":
        payload["sync"] = {"incomplete": [records[0]]}
    report = report_cls(payload)
    assert len(report.findings) == len(reasons)
    assert len({f.id for f in report.findings}) == len(reasons)
    assert {(f.status, f.kind) for f in report.findings} == {
        ("incomplete", "analysis_incomplete")
    }
    assert {f.details["reason"] for f in report.findings} == set(reasons)
    assert report.to_dict()["schema_version"] == 4
    with pytest.raises(RuntimeError):
        report.require_clean()


def test_access_pairs_keep_distinct_identities_under_data_race():
    from tirx_harness.numsim.checker_report import RaceReport

    pairs = ["write_read", "read_write", "write_write"]
    report = RaceReport({"verdict": "error", "findings": [
        {"kind": "data_race", "status": "error", "access_pair": pair}
        for pair in pairs
    ]})
    assert len({f.id for f in report.findings}) == 3
    assert {f.kind for f in report.findings} == {"data_race"}
    assert {f.details["access_pair"] for f in report.findings} == set(pairs)
    assert report.format().count("[ERROR] data_race:") == 3
