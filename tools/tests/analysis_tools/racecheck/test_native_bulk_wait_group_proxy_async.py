"""Native Racecheck ordering of `cp.async.bulk.wait_group.sync_restrict::proxy::async`.

PTX ISA 9.7.10.28.6.2: the restricted wait "only orders async-proxy global and
shared memory accesses before subsequent async-proxy accesses; not before
subsequent generic-proxy accesses". Racecheck acquires the completed groups'
causal frontier without their async-to-generic proxy bridges.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.numsim.runtime.test_bulk_wait_group_proxy_async import (
    NO_WAIT,
    PLAIN,
    PLAIN_READ,
    RESTRICTED,
    RESTRICTED_THEN_PLAIN,
    RESTRICTED_THEN_PROXY_FENCE,
    _global_read_inputs,
    _source,
    restricted_wait_keeps_newest_groups_pending,
    s2g_then_async_g2s_round_trip,
    s2g_then_generic_global_read,
    s2g_then_generic_shared_write,
)
from tirx_harness import racecheck


def _single_race(report, *, access_pair):
    native = report.native_payload
    assert report.verdict == "error", report.format()
    assert native["incomplete"] == []
    assert len(native["findings"]) == 1, native["findings"]
    finding = native["findings"][0]
    assert finding["kind"] == "data_race"
    assert finding["access_pair"] == access_pair
    return finding


def _missing_async_to_generic_bridge(finding, domain):
    assert finding["ordering_failure"] == "missing_proxy_bridge"
    assert finding["proxy_bridge"] == {
        "prior_proxy": "async",
        "current_proxy": "generic",
        "prior_domain": domain,
        "current_domain": domain,
    }


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        (PLAIN, "clean"),
        (RESTRICTED, "error"),
        (RESTRICTED_THEN_PROXY_FENCE, "clean"),
        (RESTRICTED_THEN_PLAIN, "clean"),
    ],
    ids=["plain", "restricted", "restricted-proxy-fence", "restricted-then-plain"],
)
def test_restricted_wait_does_not_order_later_generic_global_reads(mode, expected):
    report = racecheck(s2g_then_generic_global_read, _global_read_inputs(mode))
    if expected == "clean":
        report.require_clean()
        return
    finding = _single_race(report, access_pair="write_read")
    assert finding["overlap"]["byte_len"] == 16
    _missing_async_to_generic_bridge(finding, "global")


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        (PLAIN, "clean"),
        (PLAIN_READ, "clean"),
        (RESTRICTED, "error"),
        (RESTRICTED_THEN_PROXY_FENCE, "clean"),
    ],
    ids=["plain", "plain-read", "restricted", "restricted-proxy-fence"],
)
def test_restricted_wait_does_not_order_later_generic_shared_writes(mode, expected):
    inputs = {
        "mode": np.int32(mode),
        "source": _source(),
        "destination": np.zeros(16, np.uint8),
    }
    report = racecheck(s2g_then_generic_shared_write, inputs)
    if expected == "clean":
        report.require_clean()
        return
    finding = _single_race(report, access_pair="read_write")
    assert finding["overlap"]["byte_len"] == 1
    _missing_async_to_generic_bridge(finding, "shared_cta")


@pytest.mark.parametrize(
    ("mode", "expected"),
    [(PLAIN, "clean"), (RESTRICTED, "clean"), (NO_WAIT, "error")],
    ids=["plain", "restricted", "no-wait"],
)
def test_restricted_wait_orders_later_async_proxy_accesses(mode, expected):
    report = racecheck(s2g_then_async_g2s_round_trip, _global_read_inputs(mode))
    if expected == "clean":
        report.require_clean()
        return
    assert report.verdict == "error", report.format()
    findings = report.native_payload["findings"]
    assert findings
    for finding in findings:
        assert finding["kind"] == "data_race"
        assert finding["ordering_failure"] == "missing_inter_actor_sync"


@pytest.mark.parametrize("group_read", [0, 1])
def test_restricted_wait_leaves_the_newest_groups_unordered(group_read):
    inputs = {
        "group_read": np.int32(group_read),
        "source": _source(32),
        "destination": np.zeros(32, np.uint8),
        "output": np.zeros(16, np.uint8),
    }
    report = racecheck(restricted_wait_keeps_newest_groups_pending, inputs)
    if group_read == 0:
        report.require_clean()
        return
    # The newest group may still be writing what the later copy reads.
    finding = _single_race(report, access_pair="write_read")
    assert finding["ordering_failure"] == "missing_inter_actor_sync"
    assert finding["overlap"]["byte_offset"] == 16
    assert finding["overlap"]["byte_len"] == 16
