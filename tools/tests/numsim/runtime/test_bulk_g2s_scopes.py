"""Strong G2S copies retain element identity across issuing lanes and CTAs."""

import numpy as np
import pytest
import tvm
from tvm.script import tirx as T

from tirx_harness import numsim, racecheck, synccheck

from .test_bulk_copy_scopes import assert_complete_elements


def g2s_case(scope, writers="warps", cluster=False):
    actor = {"warps": "warp", "lanes": "lane", "ctas": "cta"}[writers]
    active = "lane < 2" if writers == "lanes" else "lane == 0"
    semantics = f"relaxed.{scope}" if scope else "weak"
    element = ".b128" if scope else ""
    addresses = '''
        dst = T.alloc_local((1,), "uint32")
        bar = T.alloc_local((1,), "uint32")
        T.ptx.mapa.shared__cluster.u32(dst[0], T.cuda.cvta_generic_to_shared(shared.ptr_to([actor * 4])), T.uint32(0))
        T.ptx.mapa.shared__cluster.u32(bar[0], T.cuda.cvta_generic_to_shared(barrier.ptr_to([0])), T.uint32(0))
''' if cluster else ""
    return tvm.script.from_source(
        f'''
@T.prim_func
def kernel(source: T.Buffer((16,), "uint32"), destination: T.Buffer((8,), "uint32"), enabled: T.int32):
    T.device_entry()
    _cluster = T.cluster_id([1])
    cta = T.cta_id_in_cluster([{2 if writers == "ctas" else 1}])
    warp = T.warp_id([{2 if writers == "warps" else 1}])
    lane = T.lane_id([32])
    actor = {actor}
    shared = T.alloc_shared((8,), "uint32", align=16)
    barrier = T.alloc_shared((1,), "uint64", align=16)
    if (cta == 0) & (warp == 0):
        if lane < 8:
            shared[lane] = T.uint32(0)
        if lane == 0:
            T.ptx.mbarrier.init.shared.b64(barrier.ptr_to([0]), 1)
            T.ptx.mbarrier.arrive.expect_tx.shared.b64(barrier.ptr_to([0]), T.Cast("uint32", enabled * 48))
    T.ptx.fence.proxy.async_.shared__cta()
    T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.cluster_sync()
    if {active}:
{addresses}
        T.ptx["cp.async.bulk.{semantics}.shared::{'cluster' if cluster else 'cta'}.global.mbarrier::complete_tx::bytes{element}"](
            {'dst[0]' if cluster else 'shared.ptr_to([actor * 4])'}, source.ptr_to([actor * 8]),
            T.Cast("uint32", 32 - actor * 16), {'bar[0]' if cluster else 'barrier.ptr_to([0])'}, pred=enabled != 0)
    if (cta == 0) & (warp == 0) & (lane == 0):
        T.cuda.mbarrier_wait(barrier.ptr_to([0]), 0)
    T.cuda.cluster_sync()
    if (cta == 0) & (warp == 0) & (lane < 8):
        destination[lane] = shared[lane]
''', {"T": T})


def inputs(enabled=1):
    return {
        "source": np.concatenate((np.arange(1, 9), np.arange(101, 109))).astype(np.uint32),
        "destination": np.zeros(8, np.uint32),
        "enabled": enabled,
    }


@pytest.mark.parametrize("scope", ["cta", "cluster", "gpu", "sys"])
def test_strong_g2s_scope_elements_and_predication(scope, tmp_path):
    for writers, cluster in (("warps", False), ("lanes", False), ("ctas", True)):
        kernel = g2s_case(scope, writers, cluster)
        module = numsim.transpile(kernel, cache_dir=tmp_path)
        for enabled in (0, 1):
            args = inputs(enabled)
            synccheck(kernel, args).require_clean()
            report = racecheck(kernel, args)
            if enabled and writers == "ctas" and scope == "cta":
                assert report.verdict == "error", report.format()
                assert any(f.details["access_pair"] == "write_write" for f in report.findings), report.format()
            else:
                report.require_clean()
                result = numsim.Engine().run(module, args)
                assert_complete_elements(result.outputs["destination"], enabled)


def test_weak_g2s_overlapping_writes_still_race():
    report = racecheck(g2s_case(""), inputs())
    assert report.verdict == "error", report.format()
    assert any(f.details["access_pair"] == "write_write" for f in report.findings), report.format()
