"""Native Racecheck coverage of `fence.alias.sync_restrict::read.release.sys`.

The fence releases prior reads only. Racecheck publishes it as
`fence.release.sys`, a superset of its ordering, so it reports no false race,
and records an analysis gap for every executed instance, so a write race the
superset hides leaves the verdict incomplete instead of clean.
"""

from __future__ import annotations

import pytest

from tests.numsim.runtime.test_fence_alias_read_release import (
    ALIAS_READ_RELEASE,
    _inputs,
    read_release_handoff,
)
from tirx_harness import racecheck

GAP_REASON = "read_only_release_fence_unmodeled"


@pytest.mark.parametrize(
    "publish_write", [0, 1], ids=["released-read-then-write", "unreleased-write-then-read"]
)
def test_executed_alias_read_release_fence_makes_racecheck_incomplete(publish_write):
    # Neither a false read-to-write race nor a clean verdict over the write
    # race the fence does not order: no finding, and an incomplete verdict.
    report = racecheck(read_release_handoff, _inputs(1, publish_write))
    assert report.verdict == "incomplete", report.format()
    native = report.native_payload
    assert native["findings"] == []
    gaps = [entry for entry in native["incomplete"] if entry.get("reason") == GAP_REASON]
    assert len(gaps) == 1, native["incomplete"]
    assert gaps[0]["domain"] == "memory_order"
    assert gaps[0]["effect"] == ALIAS_READ_RELEASE


def test_predicated_off_alias_read_release_fence_is_not_a_gap():
    # Without the fence nothing orders CTA 0's read before CTA 1's write.
    report = racecheck(read_release_handoff, _inputs(0))
    assert report.verdict == "error", report.format()
    native = report.native_payload
    assert native["incomplete"] == []
    assert [finding["access_pair"] for finding in native["findings"]] == ["read_write"]
