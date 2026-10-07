from __future__ import annotations

import copy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from tirx_harness._report import CheckFailed
from tirx_harness.numsim.checker_report import SyncCheckReport
from tirx_harness.numsim.checker_runner import (
    _attach_native_source_evidence_owned,
    _frontend_incomplete_payload,
    _native_diagnostic_has_source_line,
    _primfunc_source_anchor,
)
from tirx_harness.numsim.checkers import _run_synccheck as synccheck
from tirx_harness.numsim.errors import UnsupportedTIRxError
from tirx_harness.numsim.transpiler.frontend import analyze, module_spec_from_manifest
from tvm.script import tirx as T


@T.prim_func
def report_mbarrier_deadlock():
    T.device_entry()
    warp = T.warp_id([2])
    lane = T.lane_id([32])
    barriers = T.alloc_buffer((1,), "uint64", scope="shared")
    if (warp == 0) and (lane == 0):
        T.ptx.mbarrier.init.shared.b64(T.address_of(barriers[0]), 2)
        T.ptx.fence.mbarrier_init.release.cluster()
        T.ptx.mbarrier.arrive.shared.b64(T.address_of(barriers[0]))
        T.cuda.mbarrier_wait(T.address_of(barriers[0]), 0)


def _operation(
    warp_id: int,
    source_op_id: int,
    text: str,
    op_name: str,
    *,
    source_span: dict | None = None,
) -> dict:
    return {
        "kernel_index": 0,
        "global_warp_id": warp_id,
        "per_warp_sequence": 7,
        "source_op_id": source_op_id,
        "loop_frames": [{"loop_site_id": 90, "iteration_ordinal": 2}],
        "source": {
            "source_op_id": source_op_id,
            "kind": "sync",
            "op_name": op_name,
            "source_text": text,
            "source_span": source_span,
        },
    }


def _materialize_operations(value, operations):
    if isinstance(value, str) and value.startswith("$op"):
        return copy.deepcopy(operations[int(value.removeprefix("$op"))])
    if isinstance(value, dict):
        return {key: _materialize_operations(child, operations) for key, child in value.items()}
    if isinstance(value, list):
        return [_materialize_operations(child, operations) for child in value]
    return copy.deepcopy(value)


def test_cached_sync_diagnostic_reads_call_name_and_location_from_source_map() -> None:
    spec = module_spec_from_manifest(analyze(report_mbarrier_deadlock).to_manifest())
    site = next(
        entry
        for entry in spec.kernels[0].source_map
        if entry.op_name == "tirx.ptx.mbarrier_arrive_nocount"
    )
    operation = {"kernel_index": 0, "source_op_id": site.op_id}
    payload = {"verdict": "error", "findings": [{"operation": operation}]}

    _attach_native_source_evidence_owned(payload, SimpleNamespace(spec=spec))

    assert operation["source"]["op_name"] == "tirx.ptx.mbarrier_arrive_nocount"
    assert operation["source"]["kind"] == "Call"
    assert operation["source"]["source_text"] == site.text
    assert operation["source"]["source_span"]["source_name"] == str(Path(__file__).resolve())


# One case per operation-bearing container shape. Source attachment traverses
# the records, independently of diagnostic kind; this is not a serializer test.
_OPERATION_FINDING_SHAPES = [
    pytest.param(
        "findings",
        {
            "kind": "mbarrier_use_before_init",
            "operation": "$op0",
            "related_operations": ["$op1"],
        },
        id="strict_protocol",
    ),
    pytest.param(
        "findings",
        {
            "kind": "fixed_sync_protocol_error",
            "operation": "$op0",
            "related_operations": ["$op1"],
            "witness_evidence": [{"operation": "$op2"}],
        },
        id="fixed_protocol",
    ),
    pytest.param(
        "findings",
        {
            "kind": "deadlock",
            "operation": None,
            "witness_evidence": [{"operation": "$op0"}],
        },
        id="fixed_deadlock",
    ),
    pytest.param(
        "findings",
        {
            "kind": "fixed_sync_nonconfluent",
            "operation": None,
            "witnesses_evidence": [[{"operation": "$op0"}], [{"operation": "$op1"}]],
        },
        id="fixed_nonconfluent",
    ),
    pytest.param(
        "incomplete",
        {"kind": "analysis_incomplete", "reason": "effect_commit_unobserved", "operation": "$op0"},
        id="uncommitted_effect",
    ),
    pytest.param(
        "execution_error",
        {
            "kind": "deadlock",
            "blocked_operations": [{"operation": "$op0"}],
            "stalled_operations": [{"operation": "$op1"}],
        },
        id="engine_deadlock",
    ),
]


# Exercise missing/explicit-null operations, an explicit kernel, and each
# top-level diagnostic container. Unknown kinds must inherit the same fallback.
_KERNEL_ANCHORED_FINDING_SHAPES = [
    pytest.param(field, details, id=case_id)
    for field, case_id, details in (
        (
            "incomplete",
            "completion_action_unobserved",
            {"kind": "analysis_incomplete", "reason": "completion_action_unobserved"},
        ),
        (
            "incomplete",
            "cluster_barrier_warp_exit_unmodeled",
            {"kind": "analysis_incomplete", "reason": "cluster_barrier_warp_exit_unmodeled", "kernel_index": 0},
        ),
        (
            "incomplete",
            "fixed_sync_state_limit_without_operation",
            {"kind": "analysis_incomplete", "reason": "resource_limit", "resource": "fixed_sync_states", "operation": None},
        ),
        (
            "execution_error",
            "global_engine_error",
            {"kind": "engine_error", "message": "global engine failure"},
        ),
        (
            "findings",
            "future_unknown_finding_shape",
            {"kind": "future_unknown_finding_shape"},
        ),
    )
]


@pytest.mark.parametrize(
    ("field", "details"),
    [*_OPERATION_FINDING_SHAPES, *_KERNEL_ANCHORED_FINDING_SHAPES],
)
def test_every_native_finding_shape_resolves_a_parser_source_line(field, details) -> None:
    spec = analyze(report_mbarrier_deadlock)
    operations = [
        {
            "kernel_index": 0,
            "global_warp_id": index,
            "per_warp_sequence": index,
            "source_op_id": site.op_id,
            "loop_frames": [],
        }
        for index, site in enumerate(
            entry
            for entry in spec.kernels[0].source_map
            if entry.op_name
            in {
                "tirx.ptx.mbarrier_init",
                "tirx.ptx.fence_mbarrier_init",
                "tirx.ptx.mbarrier_arrive_nocount",
                "tirx.cuda.mbarrier_wait",
            }
        )
    ]
    diagnostic = _materialize_operations(details, operations)
    payload = {
        "verdict": "error" if field in {"findings", "execution_error"} else "incomplete",
        "phase": {"index": 0, "name": "source_contract"},
        "findings": [],
        "incomplete": [],
        "execution_error": None,
    }
    if field == "execution_error":
        payload[field] = diagnostic
    else:
        payload[field] = [diagnostic]

    _attach_native_source_evidence_owned(payload, SimpleNamespace(spec=spec))

    attached = payload[field] if field == "execution_error" else payload[field][0]
    assert _native_diagnostic_has_source_line(attached)
    report = SyncCheckReport.from_native(payload)
    assert report.findings
    assert all(_native_diagnostic_has_source_line(finding.details) for finding in report.findings)
    assert "  at " in report.format()


def test_clean_native_payload_skips_diagnostic_source_resolution() -> None:
    class UnexpectedSpecAccess:
        @property
        def spec(self):
            raise AssertionError("a clean report has no diagnostic source to resolve")

    payload = {
        "verdict": "clean",
        "findings": [],
        "incomplete": [],
        "execution_error": None,
        "sync": {
            "effects": [
                {
                    "kernel_index": 0,
                    "source_op_id": 1,
                }
            ]
        },
    }

    _attach_native_source_evidence_owned(payload, UnexpectedSpecAccess())

    assert "source" not in payload["sync"]["effects"][0]


@pytest.mark.parametrize("verdict", ["error", "incomplete"])
def test_synthetic_native_findings_inherit_the_kernel_source_line(verdict) -> None:
    spec = analyze(report_mbarrier_deadlock)
    payload = {
        "verdict": verdict,
        "phase": {"index": 0, "name": "source_contract"},
        "findings": [],
        "incomplete": [],
        "execution_error": None,
    }
    _attach_native_source_evidence_owned(payload, SimpleNamespace(spec=spec))

    report = SyncCheckReport.from_native(payload)

    assert len(report.findings) == 1
    assert _native_diagnostic_has_source_line(report.findings[0].details)
    assert "Kernel source: kernel 0" in report.format()
    assert "  at " in report.format()


def test_frontend_incomplete_finding_uses_the_uncompiled_primfunc_source_line() -> None:
    anchor = _primfunc_source_anchor(report_mbarrier_deadlock)
    payload = _frontend_incomplete_payload(
        "synccheck",
        UnsupportedTIRxError("unsupported for source-contract test"),
        source_anchor=anchor,
    )

    report = SyncCheckReport.from_native(payload)

    assert _native_diagnostic_has_source_line(report.findings[0].details)
    assert f"at {Path(__file__).resolve()}:" in report.format()


def test_native_race_finding_falls_back_to_kernel_name_without_parser_spans() -> None:
    spec = analyze(report_mbarrier_deadlock)
    kernel = spec.kernels[0]
    spanless_kernel = replace(
        kernel,
        source_map=tuple(replace(source, span=None) for source in kernel.source_map),
    )
    spanless_spec = replace(spec, kernels=(spanless_kernel,))
    payload = {
        "verdict": "error",
        "phase": {"index": 0, "name": "spanless"},
        "findings": [
            {
                "status": "error",
                "kind": "data_race", "access_pair": "read_write",
                "ordering_failure": "missing_proxy_bridge",
            }
        ],
        "incomplete": [],
        "execution_error": None,
    }

    _attach_native_source_evidence_owned(payload, SimpleNamespace(spec=spanless_spec))

    assert payload["findings"][0]["ordering_failure"] == "missing_proxy_bridge"
    assert payload["findings"][0]["source_anchor"] == {
        "scope": "kernel",
        "kernel_index": 0,
        "source_text": f"kernel {spanless_kernel.name}",
        "source_span": None,
    }


def test_native_finding_accepts_traced_operation_text_without_parser_spans() -> None:
    spec = analyze(report_mbarrier_deadlock)
    kernel = spec.kernels[0]
    spanless_kernel = replace(
        kernel,
        source_map=tuple(replace(source, span=None) for source in kernel.source_map),
    )
    spanless_spec = replace(spec, kernels=(spanless_kernel,))
    source = spanless_kernel.source_map[0]
    payload = {
        "verdict": "error",
        "phase": {"index": 0, "name": "spanless-traced"},
        "findings": [
            {
                "kind": "known_traced_finding",
                "operation": {"kernel_index": 0, "source_op_id": source.op_id},
            }
        ],
        "incomplete": [],
        "execution_error": None,
    }

    _attach_native_source_evidence_owned(payload, SimpleNamespace(spec=spanless_spec))

    assert payload["source_anchor"] == {
        "scope": "kernel",
        "kernel_index": 0,
        "source_text": f"kernel {spanless_kernel.name}",
        "source_span": None,
    }
    assert payload["findings"][0]["operation"]["source"] == {
        "source_op_id": source.op_id,
        "kind": source.kind,
        "source_text": source.text,
        "source_span": None,
    }


def test_typed_error_owns_source_when_execution_error_only_summarizes_abort() -> None:
    spec = analyze(report_mbarrier_deadlock)
    kernel = spec.kernels[0]
    spanless_kernel = replace(
        kernel,
        source_map=tuple(replace(source, span=None) for source in kernel.source_map),
    )
    spanless_spec = replace(spec, kernels=(spanless_kernel,))
    source = spanless_kernel.source_map[0]
    payload = {
        "verdict": "error",
        "phase": {"index": 0, "name": "spanless-typed-error"},
        "findings": [
            {
                "status": "error",
                "kind": "known_traced_finding",
                "operation": {"kernel_index": 0, "source_op_id": source.op_id},
            }
        ],
        "incomplete": [],
        "execution_error": {"kind": "engine_error", "message": "execution aborted"},
    }

    _attach_native_source_evidence_owned(payload, SimpleNamespace(spec=spanless_spec))

    assert payload["findings"][0]["operation"]["source"]["source_text"] == source.text
    assert "source_anchor" not in payload["execution_error"]


def test_native_report_renders_exact_source_and_current_execution_work() -> None:
    source = _operation(
        3,
        41,
        "T.cuda.mbarrier_wait(T.address_of(barriers[0]), 0)",
        "tirx.ptx.mbarrier_try_wait",
    )
    report = SyncCheckReport.from_native(
        {
            "verdict": "error",
            "phase": {"index": 0, "name": "wait_without_arrival"},
            "stats": {"available": True, "poll_count": 12},
            "search": {
                "algorithm": "fixed_sync_state",
                "visited_state_count": 5,
                "explored_transition_count": 8,
            },
            "findings": [
                {
                    "kind": "mbarrier_generation_release_missing",
                    "message": "the waited generation has no release",
                    "operation": source,
                }
            ],
            "incomplete": [],
            "execution_error": None,
        }
    )

    rendered = report.format()

    assert "native phase wait_without_arrival, 12 executor polls" in rendered
    assert "5 verifier states / 8 verifier transitions" in rendered
    assert "Source: warp 3, source op #41, loop iteration 2" in rendered
    assert "T.cuda.mbarrier_wait(T.address_of(barriers[0]), 0)" in rendered
    assert "0 transitions" not in rendered
    # Source spans are optional/debug metadata, not a stable report anchor.
    assert "source_span" not in rendered


def test_native_report_renders_source_excerpt_and_inline_expansion_stack(tmp_path) -> None:
    kernel_path = tmp_path / "attention.py"
    kernel_path.write_text("def kernel():\n    full.wait(stage, phase)\n")
    helper_path = tmp_path / "pipeline.py"
    helper_path.write_text("def wait(barrier):\n    T.cuda.mbarrier_wait(barrier, 0)\n")
    span = {
        "kind": "sequential",
        "spans": [
            {
                "kind": "span",
                "source_name": str(kernel_path),
                "line": 2,
                "column": 5,
                "end_line": 2,
                "end_column": 28,
            },
            {
                "kind": "span",
                "source_name": str(helper_path),
                "line": 2,
                "column": 5,
                "end_line": 2,
                "end_column": 45,
            },
        ],
    }
    report = SyncCheckReport.from_native(
        {
            "verdict": "error",
            "phase": {"index": 0, "name": "inline_wait"},
            "findings": [
                {
                    "kind": "mbarrier_generation_release_missing",
                    "message": "the waited generation has no release",
                    "operation": _operation(
                        3,
                        41,
                        "T.cuda.mbarrier_wait(barrier, 0)",
                        "tirx.ptx.mbarrier_try_wait",
                        source_span=span,
                    ),
                }
            ],
            "incomplete": [],
            "execution_error": None,
        }
    )

    rendered = report.format()

    assert f"at {kernel_path}:2:5" in rendered
    assert "2 |     full.wait(stage, phase)" in rendered
    assert "|     ^^^^^^^^^" in rendered
    assert f"expanded through {helper_path}:2:5" in rendered
    assert "TIRx: T.cuda.mbarrier_wait(barrier, 0)" in rendered

    moved_span = {
        **span,
        "spans": [{**span["spans"][0], "source_name": "/other/checkout/attention.py"}],
    }
    moved = SyncCheckReport.from_native(
        {
            "verdict": "error",
            "phase": {"index": 0, "name": "inline_wait"},
            "findings": [
                {
                    "kind": "mbarrier_generation_release_missing",
                    "message": "the waited generation has no release",
                    "operation": _operation(
                        3,
                        41,
                        "T.cuda.mbarrier_wait(barrier, 0)",
                        "tirx.ptx.mbarrier_try_wait",
                        source_span=moved_span,
                    ),
                }
            ],
            "incomplete": [],
            "execution_error": None,
        }
    )
    assert moved.findings[0].id == report.findings[0].id


def test_native_report_groups_blocked_warps_at_their_source_operation() -> None:
    blocked = []
    for warp_id in range(4, 8):
        operation = _operation(
            warp_id,
            77,
            'T.ptx.setmaxnreg(256, "inc")',
            "tirx.ptx.setmaxnreg",
        )
        blocked.append(
            {
                "warp_id": warp_id,
                "awaited_operation": "setmaxnreg.pool",
                "phase": None,
                "description": (
                    f"warp {warp_id} awaits setmaxnreg.pool key cta 0; "
                    "arrived=[4, 5, 6, 7], missing=[]"
                ),
                "operation": operation,
            }
        )
    report = SyncCheckReport.from_native(
        {
            "verdict": "error",
            "phase": {"index": 0, "name": "register_pool_deadlock"},
            "stats": {"available": True, "poll_count": 20},
            "findings": [],
            "incomplete": [],
            "execution_error": {
                "kind": "deadlock",
                "message": "executor deadlock after 20 polls",
                "blocked_operations": blocked,
                "stalled_operations": [],
            },
        }
    )

    rendered = report.format()

    assert "execution cannot make progress; blocked warps 4-7" in rendered
    assert "Blocked warps 4-7: await setmaxnreg.pool" in rendered
    assert rendered.count('T.ptx.setmaxnreg(256, "inc")') == 1
    assert "warp 4 awaits setmaxnreg.pool key" not in rendered


def test_native_report_renders_related_and_witness_source_evidence() -> None:
    report = SyncCheckReport.from_native(
        {
            "verdict": "error",
            "phase": {"index": 0, "name": "reuse"},
            "findings": [
                {
                    "kind": "fixed_sync_protocol_error",
                    "message": "generation was reused before its consumer completed",
                    "operation": _operation(
                        0,
                        10,
                        "T.ptx.mbarrier.arrive.shared.b64(T.address_of(barriers[0]))",
                        "tirx.ptx.mbarrier_arrive_nocount",
                    ),
                    "related_operations": [
                        _operation(
                            1,
                            11,
                            "T.cuda.mbarrier_wait(T.address_of(barriers[0]), 0)",
                            "tirx.ptx.mbarrier_try_wait",
                        )
                    ],
                    "witness_evidence": [
                        {
                            "operation": _operation(
                                2,
                                12,
                                "T.cuda.cta_sync()",
                                "tirx.cuda.cta_sync",
                            )
                        }
                    ],
                }
            ],
            "incomplete": [],
            "execution_error": None,
        }
    )

    rendered = report.format()

    assert "Source: warp 0" in rendered
    assert "Related: warp 1" in rendered
    assert "Witness: warp 2" in rendered


def test_real_native_deadlock_report_names_the_blocked_kernel_statement(tmp_path) -> None:
    report = synccheck(
        report_mbarrier_deadlock,
        inputs={},
        cache_dir=tmp_path,
        max_workers=1,
    )

    assert report.verdict == "error"
    assert [finding.kind for finding in report.findings] == ["deadlock"]
    rendered = report.format()
    assert "execution cannot make progress; blocked warps 0" in rendered
    assert "Blocked warps 0: await mbarrier.try_wait; phase 0" in rendered
    assert "T.cuda.mbarrier_wait(T.address_of(barriers), 0)" in rendered
    wait_line = next(
        line_number
        for line_number, line in enumerate(Path(__file__).read_text().splitlines(), start=1)
        if line.strip() == "T.cuda.mbarrier_wait(T.address_of(barriers[0]), 0)"
    )
    assert f"{Path(__file__).resolve()}:{wait_line}:" in rendered


def _review_payload(**overrides) -> dict:
    payload = {
        "verdict": "review",
        "phase": {"index": 0, "name": "advisory"},
        "stats": {"available": True, "poll_count": 1},
        "findings": [],
        "advisories": [],
        "incomplete": [],
        "execution_error": None,
    }
    payload.update(overrides)
    return payload


def test_native_synccheck_review_verdict_builds_and_renders_a_typed_advisory() -> None:
    report = SyncCheckReport.from_native(
        _review_payload(
            advisories=[
                {
                    "kind": "unresolved_completion_dependency",
                    "message": "wait cannot be proven to observe the producer's completion",
                }
            ]
        )
    )

    assert report.verdict == "review"
    assert [(finding.status, finding.kind) for finding in report.findings] == [
        ("review", "unresolved_completion_dependency")
    ]
    rendered = report.format()
    assert rendered.startswith("synccheck REVIEW - 1 finding(s);")
    assert "[REVIEW] unresolved_completion_dependency" in rendered
    assert "wait cannot be proven to observe the producer's completion" in rendered
    assert report.to_dict()["verdict"] == "review"
    assert report.to_dict()["findings"][0]["status"] == "review"


def test_native_synccheck_review_verdict_without_an_advisory_still_names_a_cause() -> None:
    report = SyncCheckReport.from_native(_review_payload())

    assert report.verdict == "review"
    assert [(finding.status, finding.kind) for finding in report.findings] == [
        ("review", "native_advisory_untyped")
    ]
    assert "[REVIEW] native_advisory_untyped" in report.format()


def test_native_synccheck_review_status_finding_is_not_reclassified_as_an_error() -> None:
    report = SyncCheckReport.from_native(
        _review_payload(
            findings=[
                {
                    "kind": "same_warp_register_dependency",
                    "status": "review",
                    "message": (
                        "resolution depends on a register dependency synccheck does not model"
                    ),
                }
            ]
        )
    )

    assert report.verdict == "review"
    assert [finding.status for finding in report.findings] == ["review"]
    with pytest.raises(CheckFailed, match="synccheck review"):
        report.require_clean()


def test_native_synccheck_findings_without_a_status_stay_errors() -> None:
    report = SyncCheckReport.from_native(
        {
            "verdict": "error",
            "phase": {"index": 0, "name": "typed_error"},
            "findings": [{"kind": "fixed_sync_protocol_error", "message": "arrive over-counts"}],
            "incomplete": [],
            "execution_error": None,
        }
    )

    assert [finding.status for finding in report.findings] == ["error"]
