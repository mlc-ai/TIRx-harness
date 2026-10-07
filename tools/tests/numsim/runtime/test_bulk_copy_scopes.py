"""Strong bulk copies are atomic per b128 element, not per whole window."""

import numpy as np
import pytest
import tvm
from tvm.script import tirx as T

from tirx_harness import numsim, racecheck, synccheck


def copy_case(scope, relation="cta"):
    semantics = f"relaxed.{scope}" if scope else "weak"
    element = ".b128" if scope else ""

    return tvm.script.from_source(
        f'''
@T.prim_func
def kernel(destination: T.Buffer((8,), "uint32"), enabled: T.int32):
    T.device_entry()
    cluster = T.cluster_id([{2 if relation == "gpu" else 1}])
    cta = T.cta_id_in_cluster([{2 if relation == "cluster" else 1}])
    warp = T.warp_id([{2 if relation == "cta" else 1}])
    lane = T.lane_id([32])
    actor = cluster + cta + warp
    shared = T.alloc_shared((16,), "uint32", align=16)
    if lane < 8:
        shared[warp * 8 + lane] = T.Cast("uint32", actor * 100 + lane + 1)
    T.cuda.warp_sync()
    T.ptx.fence.proxy.async_.shared__cta()
    T.ptx["cp.async.bulk.{semantics}.global.shared::cta.bulk_group{element}"](
        destination.ptr_to([actor * 4]), shared.ptr_to([warp * 8]),
        T.Cast("uint32", 32 - actor * 16), pred=(lane == 0) & (enabled != 0))
    if lane == 0:
        T.ptx.cp.async_.bulk.commit_group()
        T.ptx.cp.async_.bulk.wait_group.read(0)
    T.cuda.warp_sync()
    if lane < 8:
        shared[warp * 8 + lane] = T.uint32(999)
    T.cuda.warp_sync()
    if lane == 0:
        T.ptx.cp.async_.bulk.wait_group(0)
''',
        {"T": T},
    )


def assert_complete_elements(output, enabled):
    if not enabled:
        np.testing.assert_array_equal(output, np.zeros(8, np.uint32))
        return
    np.testing.assert_array_equal(output[:4], np.arange(1, 5))
    assert np.array_equal(output[4:], np.arange(5, 9)) or np.array_equal(
        output[4:], np.arange(101, 105)
    ), output


@pytest.mark.parametrize("scope", ["cta", "cluster", "gpu", "sys"])
def test_bulk_copy_scope_and_elements(scope, tmp_path):
    for relation in ("cta", "cluster", "gpu"):
        kernel = copy_case(scope, relation)
        module = numsim.transpile(kernel, cache_dir=tmp_path)
        for enabled in (0, 1):
            args = {"destination": np.zeros(8, np.uint32), "enabled": enabled}
            synccheck(kernel, args).require_clean()
            report = racecheck(kernel, args)
            mismatch = enabled and (
                (scope == "cta" and relation != "cta") or (scope == "cluster" and relation == "gpu")
            )
            if mismatch:
                assert report.verdict == "error", report.format()
                assert any(f.kind == "scope_mismatch" for f in report.findings), report.format()
            else:
                report.require_clean()
                result = numsim.Engine().run(module, args)
                assert_complete_elements(result.outputs["destination"], enabled)


def test_bulk_copy_weak_overlap_still_races():
    kernel = copy_case("")
    args = {"destination": np.zeros(8, np.uint32), "enabled": 1}
    synccheck(kernel, args).require_clean()
    report = racecheck(kernel, args)
    assert report.verdict == "error", report.format()
    assert any(f.details["access_pair"] == "write_write" for f in report.findings), report.format()
