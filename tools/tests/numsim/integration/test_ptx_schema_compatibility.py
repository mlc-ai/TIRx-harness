"""Absent optional collector slots retain ordinary MMA semantics."""

from dataclasses import replace

import pytest

from tests.numsim.microtests.cases.tcgen05_mma_forms import (
    _make_raw_f16_arguments,
    raw_f16_ss_m64_layout_f_valid_descriptor,
)
from tirx_harness import racecheck, synccheck
from tirx_harness.numsim.transpiler import build
from tirx_harness.numsim.transpiler import ptx_dialect


@pytest.mark.parametrize("checker", (synccheck, racecheck), ids=("synccheck", "racecheck"))
def test_dense_mma_missing_collector_slot(checker, monkeypatch, tmp_path):
    # Force the schema-compatibility path even if this kernel was previously
    # cached. Removing an absent optional field does not change its semantics.
    monkeypatch.setenv("NUMSIM_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(build, "load_cached_generated_artifact", lambda _prepared: None)
    decode = ptx_dialect.decode_ptx_call
    seen = []

    def with_collector_slot(call):
        decoded = decode(call)
        if decoded.op_name == "tirx.ptx.tcgen05_mma_ss":
            assert not decoded.modifiers.get("collector_a", "")
            seen.append(decoded.op_name)
            modifiers = {k: v for k, v in decoded.modifiers.items() if k != "collector_a"}
            decoded = replace(decoded, modifiers=modifiers)
        return decoded

    monkeypatch.setattr(ptx_dialect, "decode_ptx_call", with_collector_slot)
    report = checker(raw_f16_ss_m64_layout_f_valid_descriptor, _make_raw_f16_arguments())
    assert seen
    report.require_clean()
