"""Source-evidence rendering for native checker findings."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from tirx_harness._report import Finding


_DYNAMIC_SEQUENCE_TEXT = re.compile(r"(?<=/seq:)\d+")
_NATIVE_PRESENTATION_FIELDS = frozenset(
    {"source_anchor", "source_location", "source_span", "source_excerpt"}
)


def _native_finding_identity(value: Any) -> Any:
    """Remove scheduler-local counters from a native finding's stable identity."""

    if isinstance(value, dict):
        return {
            key: _native_finding_identity(item)
            for key, item in value.items()
            if key != "per_warp_sequence" and key not in _NATIVE_PRESENTATION_FIELDS
        }
    if isinstance(value, list):
        return [_native_finding_identity(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_native_finding_identity(item) for item in value)
    if isinstance(value, str):
        return _DYNAMIC_SEQUENCE_TEXT.sub("*", value)
    return value


def _native_int_ranges(values: list[int]) -> str:
    """Render sorted integer IDs compactly without hiding any participants."""

    ordered = sorted(set(values))
    if not ordered:
        return "none"
    ranges: list[str] = []
    start = previous = ordered[0]
    for value in ordered[1:]:
        if value == previous + 1:
            previous = value
            continue
        ranges.append(str(start) if start == previous else f"{start}-{previous}")
        start = previous = value
    ranges.append(str(start) if start == previous else f"{start}-{previous}")
    return ", ".join(ranges)


def _native_operation(value: Any) -> dict[str, Any] | None:
    """Return a dynamic operation record from a direct or wrapped payload value."""

    if not isinstance(value, dict):
        return None
    operation = value.get("operation")
    if isinstance(operation, dict):
        return operation
    if isinstance(value.get("source_op_id"), int):
        return value
    return None


def _native_nested_operations(value: Any):
    """Yield every dynamic operation in an arbitrary diagnostic payload shape."""

    if isinstance(value, dict):
        if (
            isinstance(value.get("kernel_index"), int)
            and not isinstance(value.get("kernel_index"), bool)
            and isinstance(value.get("source_op_id"), int)
            and not isinstance(value.get("source_op_id"), bool)
        ):
            yield value
        for key, child in value.items():
            if key not in {"source", "source_anchor"}:
                yield from _native_nested_operations(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from _native_nested_operations(child)


def _native_operation_identity(operation: dict[str, Any]) -> tuple[Any, ...]:
    source = operation.get("source")
    source_text = source.get("source_text") if isinstance(source, dict) else None
    return (
        operation.get("kernel_index"),
        operation.get("global_warp_id"),
        operation.get("source_op_id"),
        tuple(
            (frame.get("loop_site_id"), frame.get("iteration_ordinal"))
            for frame in operation.get("loop_frames", ())
            if isinstance(frame, dict)
        ),
        source_text,
    )


def _native_operation_text(operation: dict[str, Any], *, include_warp: bool = True) -> str:
    """Render one exact source operation using stable runtime/source-map evidence."""

    source = operation.get("source") if isinstance(operation.get("source"), dict) else {}
    source_text = source.get("source_text")
    op_name = source.get("op_name")
    if not op_name and source.get("kind") not in {None, "Call"}:
        op_name = source.get("kind")
    location: list[str] = []
    warp_id = operation.get("global_warp_id")
    if include_warp and isinstance(warp_id, int):
        location.append(f"warp {warp_id}")
    source_op_id = operation.get("source_op_id")
    if isinstance(source_op_id, int):
        location.append(f"source op #{source_op_id}")
    loop_frames = operation.get("loop_frames")
    if isinstance(loop_frames, list) and loop_frames:
        iterations = [
            str(frame.get("iteration_ordinal"))
            for frame in loop_frames
            if isinstance(frame, dict) and isinstance(frame.get("iteration_ordinal"), int)
        ]
        if iterations:
            location.append(f"loop iteration {'/'.join(iterations)}")
    prefix = ", ".join(location) or "unknown dynamic operation"
    if isinstance(source_text, str) and source_text:
        kind = f" [{op_name}]" if isinstance(op_name, str) and op_name else ""
        return f"{prefix}: {source_text}{kind}"
    return prefix


def _native_source_span_leaves(value: Any) -> tuple[dict[str, Any], ...]:
    """Flatten one structured Span/SequentialSpan payload in provenance order."""

    if not isinstance(value, dict):
        return ()
    kind = value.get("kind")
    if kind == "span":
        source_name = value.get("source_name")
        coordinates = tuple(
            value.get(name) for name in ("line", "column", "end_line", "end_column")
        )
        if (
            isinstance(source_name, str)
            and source_name
            and all(
                isinstance(item, int) and not isinstance(item, bool) and item >= 1
                for item in coordinates
            )
            and coordinates[2] >= coordinates[0]
            and (coordinates[2] != coordinates[0] or coordinates[3] >= coordinates[1])
        ):
            return (value,)
        return ()
    if kind != "sequential" or not isinstance(value.get("spans"), list):
        return ()
    return tuple(leaf for child in value["spans"] for leaf in _native_source_span_leaves(child))


def _native_source_location(span: dict[str, Any]) -> str:
    start = f"{span['source_name']}:{span['line']}:{span['column']}"
    if span["end_line"] == span["line"]:
        return start
    return f"{start}-{span['end_line']}:{span['end_column']}"


def _native_source_excerpt(span: dict[str, Any], *, max_lines: int = 5) -> list[str]:
    """Render a bounded parser-style excerpt, or no lines when source is unavailable."""

    source_name = span["source_name"]
    if source_name.startswith("<") and source_name.endswith(">"):
        return []
    try:
        source_lines = Path(source_name).read_text(errors="replace").splitlines()
    except OSError:
        return []
    start = span["line"]
    end = span["end_line"]
    if start < 1 or end < start or end > len(source_lines):
        return []
    selected = list(range(start, min(end, start + max_lines - 1) + 1))
    truncated = selected[-1] < end
    width = len(str(selected[-1]))
    rendered: list[str] = []
    for line_number in selected:
        text = source_lines[line_number - 1]
        first_column = span["column"] if line_number == start else 1
        last_column = span["end_column"] if line_number == end else len(text) + 1
        first_column = min(max(first_column, 1), len(text) + 1)
        last_column = min(max(last_column, first_column + 1), len(text) + 1)
        caret_count = max(1, last_column - first_column)
        rendered.append(f"{line_number:>{width}} | {text}")
        rendered.append(f"{'':>{width}} | {' ' * (first_column - 1)}{'^' * caret_count}")
    if truncated:
        rendered.append(f"{'':>{width}} | ...")
    return rendered


def _native_operation_evidence_lines(
    operation: dict[str, Any],
    *,
    label: str,
    indent: str,
    include_warp: bool = True,
) -> list[str]:
    """Render dynamic context, structured source location, and lowered TIRx evidence."""

    source = operation.get("source") if isinstance(operation.get("source"), dict) else {}
    spans = _native_source_span_leaves(source.get("source_span"))
    if not spans:
        return [f"{indent}{label}: {_native_operation_text(operation, include_warp=include_warp)}"]

    context: list[str] = []
    if operation.get("source_scope") == "kernel":
        kernel_index = operation.get("kernel_index")
        context.append(f"kernel {kernel_index}" if isinstance(kernel_index, int) else "kernel")
    warp_id = operation.get("global_warp_id")
    if include_warp and isinstance(warp_id, int):
        context.append(f"warp {warp_id}")
    source_op_id = operation.get("source_op_id")
    if isinstance(source_op_id, int):
        context.append(f"source op #{source_op_id}")
    loop_frames = operation.get("loop_frames")
    if isinstance(loop_frames, list) and loop_frames:
        iterations = [
            str(frame.get("iteration_ordinal"))
            for frame in loop_frames
            if isinstance(frame, dict) and isinstance(frame.get("iteration_ordinal"), int)
        ]
        if iterations:
            context.append(f"loop iteration {'/'.join(iterations)}")
    lines = [f"{indent}{label}: {', '.join(context) or 'unknown dynamic operation'}"]

    primary, *expansions = spans
    lines.append(f"{indent}  at {_native_source_location(primary)}")
    lines.extend(f"{indent}    {line}" for line in _native_source_excerpt(primary))
    for expansion in expansions:
        lines.append(f"{indent}  expanded through {_native_source_location(expansion)}")

    source_text = source.get("source_text")
    op_name = source.get("op_name")
    if not op_name and source.get("kind") not in {None, "Call"}:
        op_name = source.get("kind")
    if isinstance(source_text, str) and source_text:
        kind = f" [{op_name}]" if isinstance(op_name, str) and op_name else ""
        lines.append(f"{indent}  TIRx: {source_text}{kind}")
    return lines


def _native_blocked_operation_lines(details: dict[str, Any]) -> list[str]:
    """Group equivalent blocked warps while retaining their exact source operation."""

    blocked = details.get("blocked_operations")
    if not isinstance(blocked, list) or not blocked:
        return []
    groups: dict[tuple[Any, ...], dict[str, Any]] = {}
    for item in blocked:
        if not isinstance(item, dict):
            continue
        operation = _native_operation(item)
        awaited = item.get("awaited_operation")
        phase = item.get("phase")
        description = item.get("description")
        participant = None
        if isinstance(description, str) and "; " in description:
            participant = description.split("; ", 1)[1]
        key = (
            awaited,
            phase,
            participant,
            None if operation is None else _native_operation_identity(operation)[2:],
        )
        group = groups.setdefault(
            key,
            {
                "warps": [],
                "awaited": awaited,
                "phase": phase,
                "participant": participant,
                "operation": operation,
            },
        )
        warp_id = item.get("warp_id")
        if isinstance(warp_id, int):
            group["warps"].append(warp_id)

    lines: list[str] = []
    for group in groups.values():
        warps = _native_int_ranges(group["warps"])
        awaited = group["awaited"] or "an unresolved synchronization operation"
        suffix: list[str] = []
        if group["phase"] is not None:
            suffix.append(f"phase {group['phase']}")
        if group["participant"]:
            suffix.append(str(group["participant"]))
        extra = f"; {'; '.join(suffix)}" if suffix else ""
        lines.append(f"    Blocked warps {warps}: await {awaited}{extra}")
        operation = group["operation"]
        if operation is not None:
            lines.extend(
                _native_operation_evidence_lines(
                    operation, label="Source", indent="      ", include_warp=False
                )
            )
    return lines


def _native_stalled_operation_lines(details: dict[str, Any]) -> list[str]:
    stalled = details.get("stalled_operations")
    if not isinstance(stalled, list) or not stalled:
        return []
    groups: dict[tuple[Any, ...], dict[str, Any]] = {}
    for item in stalled:
        operation = _native_operation(item)
        if operation is None:
            continue
        key = _native_operation_identity(operation)[2:]
        group = groups.setdefault(key, {"warps": [], "operation": operation})
        warp_id = item.get("warp_id") if isinstance(item, dict) else None
        if isinstance(warp_id, int):
            group["warps"].append(warp_id)
    lines: list[str] = []
    for group in groups.values():
        lines.append(f"    Stalled warps {_native_int_ranges(group['warps'])}:")
        lines.extend(
            _native_operation_evidence_lines(
                group["operation"], label="Source", indent="      ", include_warp=False
            )
        )
    return lines


def _native_finding_evidence_lines(finding: Finding) -> list[str]:
    """Render the source and causal evidence an agent needs to act on a finding."""

    details = finding.details
    lines: list[str] = []
    seen: set[tuple[Any, ...]] = set()

    def add_operation(label: str, value: Any) -> None:
        operation = _native_operation(value)
        if operation is None:
            return
        identity = _native_operation_identity(operation)
        if identity in seen:
            return
        seen.add(identity)
        lines.extend(_native_operation_evidence_lines(operation, label=label, indent="    "))

    add_operation("Source", details.get("operation"))
    related_operations = details.get("related_operations")
    if isinstance(related_operations, list):
        for value in related_operations:
            add_operation("Related", value)
    for label, key in (("Prior", "prior"), ("Current", "current")):
        add_operation(label, details.get(key))
    for label, key in (("Reader", "reader_operation"), ("Writer", "writer_operation")):
        add_operation(label, details.get(key))
    for label, key in (("Wait", "wait_operation"), ("Plain access", "plain_operation")):
        add_operation(label, details.get(key))
    witness_evidence = details.get("witness_evidence")
    if isinstance(witness_evidence, list):
        for value in witness_evidence:
            add_operation("Witness", value)

    lines.extend(_native_blocked_operation_lines(details))
    lines.extend(_native_stalled_operation_lines(details))

    handled = {
        "operation",
        "related_operations",
        "prior",
        "current",
        "reader_operation",
        "writer_operation",
        "wait_operation",
        "plain_operation",
        "witness_evidence",
        "blocked_operations",
        "stalled_operations",
        "source_anchor",
    }
    for key, value in details.items():
        if key in handled:
            continue
        label = (
            "Witness"
            if key == "witnesses_evidence"
            else {
                "cta_sync_operation": "CTA sync",
                "named_barrier_operation": "Named barrier",
            }.get(key, "Related")
        )
        for operation in _native_nested_operations(value):
            add_operation(label, operation)

    source_anchor = details.get("source_anchor")
    if isinstance(source_anchor, dict):
        lines.extend(
            _native_operation_evidence_lines(
                {
                    "kernel_index": source_anchor.get("kernel_index"),
                    "source_scope": "kernel",
                    "source": source_anchor,
                },
                label="Kernel source",
                indent="    ",
                include_warp=False,
            )
        )
    return lines


def _native_finding_headline(finding: Finding) -> str:
    details = finding.details
    blocked = details.get("blocked_operations")
    stalled = details.get("stalled_operations")
    if finding.kind == "deadlock" or finding.kind.endswith("_deadlock"):
        warp_ids = [
            item.get("warp_id")
            for group in (blocked, stalled)
            if isinstance(group, list)
            for item in group
            if isinstance(item, dict) and isinstance(item.get("warp_id"), int)
        ]
        if warp_ids:
            return f"execution cannot make progress; blocked warps {_native_int_ranges(warp_ids)}"
    return (str(finding.message).splitlines() or [""])[0]


def _native_execution_label(payload: dict[str, Any]) -> str:
    """Describe current native work without referring to removed replay traces."""

    parts: list[str] = []
    stats = payload.get("stats")
    if isinstance(stats, dict) and stats.get("available", True):
        poll_count = stats.get("poll_count")
        if isinstance(poll_count, int):
            parts.append(f"{poll_count} executor polls")
    search = payload.get("search")
    if isinstance(search, dict) and search.get("algorithm") == "fixed_sync_state":
        states = search.get("visited_state_count")
        transitions = search.get("explored_transition_count")
        if (
            isinstance(states, int)
            and isinstance(transitions, int)
            and (states > 0 or transitions > 0)
        ):
            parts.append(f"{states} verifier states / {transitions} verifier transitions")
    return ", ".join(parts) or "direct native execution"
