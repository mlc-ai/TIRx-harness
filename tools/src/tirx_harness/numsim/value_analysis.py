"""Default, conservative value-determinism policy for native Racecheck.

An unordered atomic load with distinct candidate words is an error, even if
ordinary control flow or downstream computation would discard the difference.
Only explicit ld_until sites may summarize retry values by their exit values.
Concurrent RMW ordering is also rejected; matching output from one alternative
execution is not evidence that every returned ticket or result is fixed.
"""

from __future__ import annotations

from typing import Any


def attach_value_analysis(payload: dict[str, Any], module: Any) -> dict[str, Any]:
    """Annotate an owned native payload before source evidence is attached."""
    evidence: dict[str, Any] = {"proven_poll_exits": [], "fixed_reads": []}
    payload["value_analysis"] = evidence
    if payload.get("verdict") not in {"clean", "review"}:
        evidence["skipped"] = "baseline_verdict"
        return payload

    poll_sites = {
        (index, source.op_id)
        for index, kernel in enumerate(module.spec.kernels)
        for source in kernel.source_map
        if getattr(getattr(source.node, "op", None), "name", None) == "tirx.cuda.ld_until"
    }
    findings = payload.setdefault("findings", [])
    incomplete = payload.setdefault("incomplete", [])
    for observation in payload.get("observations", ()):
        operation = observation.get("read_operation") or {}
        is_poll = (operation.get("kernel_index"), observation.get("source_op_id")) in poll_sites
        summary = {
            key: observation.get(key)
            for key in (
                "space",
                "read_operation",
                "source_op_id",
                "global_warp_id",
                "lane",
                "span",
                "candidates",
                "occurrences",
            )
        }
        summary["primitive"] = "ld_until" if is_poll else None
        candidates = observation.get("candidates") or []
        values = {
            candidate["value_hex"]
            for candidate in candidates
            if candidate.get("value_hex") is not None
        }
        complete = (
            bool(candidates)
            and not observation.get("truncated")
            and all(candidate.get("value_hex") is not None for candidate in candidates)
        )
        exits = observation.get("poll_exit_values")
        # A complete unique exit is a value proof only for the explicit wait.
        # The native evaluator currently models relaxed waits; acquire effects
        # cannot be summarized by their final word alone.
        if is_poll and complete and exits is not None and len(exits) == 1:
            evidence["proven_poll_exits"].append({**summary, "exit_value_hex": exits[0]})
            continue
        if not is_poll and complete and len(values) == 1:
            evidence["fixed_reads"].append(summary)
            continue

        uncertain = len(exits or ()) > 1 if is_poll else len(values) > 1
        if uncertain:
            reason = "multiple_poll_exits" if is_poll else "unordered_atomic_read"
            message = (
                "ld_until can exit with different values"
                if is_poll
                else "atomic load has multiple possible values; polling must use ld_until "
                "with a unique exit value, not an ordinary while loop"
            )
            findings.append(
                {
                    "kind": "schedule_dependent_atomic_value",
                    "status": "error",
                    "reason": reason,
                    "message": message,
                    "site": summary,
                    "values_hex": sorted(exits if is_poll else values),
                }
            )
        else:
            reason = (
                "chain_truncated"
                if observation.get("truncated")
                else ("poll_exit_unmodeled" if is_poll else "candidate_values_unavailable")
            )
            incomplete.append(
                {
                    "kind": "value_analysis_incomplete",
                    "reason": reason,
                    "message": f"cannot establish a fixed value at source op {summary['source_op_id']}: {reason}",
                    "site": summary,
                }
            )

    for group in payload.get("rmw_groups", ()):
        # The group itself is a native witness of concurrent atomic modification
        # order. Do not infer determinism from unused returns or one final sum.
        findings.append(
            {
                "kind": "schedule_dependent_atomic_value",
                "status": "error",
                "reason": "concurrent_rmw_order",
                "message": "concurrent atomic RMW order is not fixed; returned values or accumulated results may depend on scheduling",
                "group": group,
            }
        )

    if any(finding.get("status", "error") == "error" for finding in findings):
        payload["verdict"] = "error"
    elif incomplete:
        payload["verdict"] = "incomplete"
    return payload
