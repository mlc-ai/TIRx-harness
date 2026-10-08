from __future__ import annotations

import numpy as np
import pytest
from tvm.script import tirx as T

from tirx_harness import numsim


@T.prim_func
def profiled_branch(
    source: T.Buffer((64,), "uint32"),
    output: T.Buffer((64,), "uint32"),
    choose_add: T.int32,
):
    T.device_entry()
    cta = T.cta_id([2])
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    index = cta * 32 + lane
    seen = T.alloc_local((1,), "uint32")
    if choose_add != 0:
        T.cuda.wait_until(seen[0], source.ptr_to([index]), seen[0] != 0, "gpu", "global")
        T.ptx.add.u32(output[index], seen[0], T.uint32(1))
    else:
        T.ptx.sub.u32(output[index], source[index], T.uint32(1))
    T.ptx["bar.warp.sync"](T.uint32(0xFFFFFFFF))


def assert_executed_branch(kernel_spec, records, choose_add):
    pairs = [(entry["site_id"], entry["variant"]) for entry in records]
    assert pairs == sorted(set(pairs))
    sites = {}
    for operation in ("add", "sub"):
        matched = [
            entry.op_id
            for entry in kernel_spec.source_map
            if entry.kind == "Call" and entry.op_name == f"tirx.ptx.{operation}_int"
        ]
        assert len(matched) == 1
        sites[operation] = matched[0]
    selected, absent = ("add", "sub") if choose_add else ("sub", "add")
    assert any(site == sites[selected] and name.endswith("::U32") for site, name in pairs)
    assert not any(site == sites[absent] for site, _ in pairs)
    barrier_sites = [
        entry.op_id for entry in kernel_spec.source_map if entry.op_name == "tirx.ptx.bar_warp_sync"
    ]
    assert len(barrier_sites) == 1
    assert any(
        site == barrier_sites[0] and name.endswith("::bar_warp_sync") for site, name in pairs
    )
    wait_sites = [
        entry.op_id for entry in kernel_spec.source_map if entry.op_name == "tirx.cuda.wait_until"
    ]
    assert len(wait_sites) == 1
    wait_variants = [
        name for site, name in pairs if site == wait_sites[0] and "::Acquire<" in name
    ]
    assert bool(wait_variants) == bool(choose_add)
    assert all("::U32" in name and "::Global" in name and "::Gpu" in name for name in wait_variants)


@pytest.mark.parametrize("profiled", [False, True])
def test_instruction_profiles_keep_kernel_sites_and_runtime_branches_separate(
    monkeypatch, tmp_path, profiled
):
    monkeypatch.setenv("NUMSIM_PROFILE", "1" if profiled else "0")
    module = numsim.transpile([profiled_branch, profiled_branch], cache_dir=tmp_path)
    source = np.arange(64, dtype=np.uint32) + 10
    args = {
        f"k{phase}:{name}": value
        for phase in range(2)
        for name, value in {
            "source": source,
            "output": np.zeros(64, dtype=np.uint32),
            "choose_add": phase,
        }.items()
    }
    engine = numsim.Engine(max_workers=2)
    for _ in range(2):
        result = engine.run(module, args, outputs=("k0:output", "k1:output"))
        for phase, stats in enumerate(result.stats["kernels"]):
            assert stats["kernel_index"] == phase
            np.testing.assert_array_equal(
                result.outputs[f"k{phase}:output"], source + 1 if phase else source - 1
            )
            if profiled:
                assert_executed_branch(
                    module.spec.kernels[phase], stats["executed_instruction_variants"], phase
                )
            else:
                assert "executed_instruction_variants" not in stats
