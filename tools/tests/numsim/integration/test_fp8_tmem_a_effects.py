"""FP8 TS shares the numerical path and must retain the TMEM read lifetime."""

import pytest

from tests.numsim.microtests.cases.tcgen05_fp8_ts import fp8_ts_arguments, fp8_ts_kernel
from tirx_harness import racecheck, synccheck


@pytest.mark.parametrize("m,d_f16", [(64, False), (128, True)])
def test_fp8_tmem_a_requires_published_stores(m, d_f16):
    arguments, _ = fp8_ts_arguments(m, d_f16)
    for checker in (synccheck, racecheck):
        checker(fp8_ts_kernel(m, d_f16), arguments).require_clean()
    # A memory-lifetime conflict belongs to Racecheck, not Synccheck.
    report = racecheck(fp8_ts_kernel(m, d_f16, publish=False), arguments)
    assert report.verdict == "error", report.format()
    (finding,) = report.to_dict()["findings"]
    assert (finding["status"], finding["kind"]) == ("error", "data_race")
    assert finding["details"]["access_pair"] == "write_read"
    witness = finding["details"]
    assert witness["ordering_failure"] == "async_lifetime_not_drained"
    assert witness["prior"]["space"] == witness["current"]["space"] == "tmem"
    assert witness["overlap"]["byte_offset"] == 16 * 4
    assert witness["overlap"]["byte_len"] == 8 * 4
    assert "kind::f8f6f4" in witness["current"]["operation"]["source"]["source_text"]
