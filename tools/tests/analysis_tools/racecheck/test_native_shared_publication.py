"""Global publication transfers shared history only between synchronized lanes."""

import numpy as np
import pytest
import tvm
from tvm.script import tirx as T

from tirx_harness import numsim, racecheck, synccheck


def _check_race_report(report, expected, space="shared"):
    assert report.verdict == expected, report.format()
    if expected == "error":
        assert report.findings, report.format()
        for finding in report.findings:
            assert finding.status == "error" and finding.details["access_pair"] in {"write_read", "read_write"}, report.format()
            for side in ("prior", "current"):
                witness = finding.details[side]
                assert witness["space"] == space, report.format()
                assert witness["operation"]["source"]["source_text"].strip(), report.format()


@pytest.mark.parametrize("relay", ("warp", "partial_warp", "named", "release", "relaxed"))
def test_shared_frontier_relay(relay, tmp_path):
    warp_relay = relay in ("warp", "partial_warp")
    if warp_relay:
        finish = """
    if warp == 1:
        T.cuda.warp_sync()
        if lane == 1:
            output[0] = data[0]
"""
        if relay == "partial_warp":
            finish = finish.replace(
                "T.cuda.warp_sync()",
                'T.ptx["bar.warp.sync"](T.uint32(6), pred=(lane == 1) | (lane == 2))',
            )
        publication = ""
    elif relay == "named":
        finish = """
    if warp >= 1:
        T.ptx.bar.sync(T.uint32(6), T.uint32(64))
    if (warp == 2) and (lane == 0):
        output[0] = data[0]
"""
        publication = ""
    else:
        publication = f'T.ptx["st.{relay}.cta.global.s32"](flag.ptr_to([1]), T.int32(1))'
        finish = """
    if (warp == 2) and (lane == 0):
        observed = 0
        T.cuda.wait_until(
            observed, flag.ptr_to([1]), observed != 0,
            scope="cta", ptx_type="s32",
        )
        output[0] = data[0]
"""
    kernel = tvm.script.from_source(
        f"""@T.prim_func
def kernel(flag: T.Buffer((2,), "int32"), output: T.Buffer((1,), "int32")):
    T.device_entry()
    warp = T.warp_id([{2 if warp_relay else 3}])
    lane = T.lane_id([32])
    data = T.alloc_shared((1,), "int32")
    observed = T.local_scalar("int32")
    if (warp == 0) and (lane == 0):
        data[0] = 41
        T.ptx.st.release.cta.global_.s32(flag.ptr_to([0]), T.int32(1))
    if (warp == 1) and (lane == 0):
        observed = 0
        while observed == 0:
            T.ptx.ld.acquire.cta.global_.s32(observed, flag.ptr_to([0]))
        {publication}
    {finish}
""",
        {"T": T},
    )
    inputs = dict(flag=np.zeros(2, np.int32), output=np.zeros(1, np.int32))
    synccheck(kernel, inputs).require_clean()
    report = racecheck(kernel, inputs)
    expected = "error" if relay in ("partial_warp", "relaxed") else "clean"
    _check_race_report(report, expected)
    if expected == "clean":
        result = numsim.Engine().run(
            numsim.transpile(kernel, cache_dir=tmp_path, _analysis_checker=None), inputs
        )
        np.testing.assert_array_equal(result.outputs["output"], np.array([41], np.int32))


@pytest.mark.parametrize("fence_side", ("producer", "consumer", "early_producer", "early_consumer"))
def test_shared_async_handoff(fence_side, tmp_path):
    controls = (
        ((0, 0, "error"),)
        if fence_side.startswith("early")
        else ((0, 0, "clean"), (1, 0, "error"), (0, 1, "error"))
    )
    for writer, reader, expected in controls:
        fence = "T.ptx.fence.proxy.async_.shared__cta()"
        kernel = tvm.script.from_source(
            f"""@T.prim_func
def kernel(flag: T.Buffer((1,), "int32"), output: T.Buffer((4,), "int32")):
    T.device_entry()
    warp = T.warp_id([2])
    lane = T.lane_id([32])
    data = T.alloc_shared((4,), "int32", align=16)
    observed = T.local_scalar("int32")
    if warp == 0:
        if lane == 0:
            {fence if fence_side == "early_producer" else "pass"}
        if lane == {writer}:
            for i in T.serial(4):
                data[i] = 41 + i
        if lane == 0:
            {fence if fence_side == "producer" else ""}
            T.ptx.st.release.cta.global_.s32(flag.ptr_to([0]), T.int32(1))
    else:
        if lane == 0:
            {fence if fence_side == "early_consumer" else ""}
            observed = 0
            while observed == 0:
                T.ptx.ld.acquire.cta.global_.s32(observed, flag.ptr_to([0]))
        if lane == {reader}:
            {fence if fence_side == "consumer" else ""}
            T.ptx["cp.async.bulk.global.shared::cta.bulk_group"](
                output.ptr_to([0]), data.ptr_to([0]), T.uint32(16))
            T.ptx.cp.async_.bulk.commit_group()
            T.ptx.cp.async_.bulk.wait_group(0)
""",
            {"T": T},
        )
        inputs = dict(flag=np.zeros(1, np.int32), output=np.zeros(4, np.int32))
        synccheck(kernel, inputs).require_clean()
        report = racecheck(kernel, inputs)
        _check_race_report(report, expected)
        if expected == "clean":
            actual = numsim.Engine().run(
                numsim.transpile(kernel, cache_dir=tmp_path, _analysis_checker=None), inputs
            )
            np.testing.assert_array_equal(actual.outputs["output"], np.arange(4) + 41)


def _fence_case(space, order, publisher=0, consumer=1):
    allocation = 'shared = T.alloc_shared((1,), "int32")' if space == "shared" else ""
    data = "shared" if space == "shared" else "data"
    mnemonic = f"fence{'.' + order if order else ''}.cta"
    publication = (
        f"if lane == {publisher}:\n        T.cuda.thread_fence()"
        if order == "cuda"
        else f'T.ptx["{mnemonic}"](pred=lane == {publisher})'
    )
    acquisition = (
        f"if lane == {consumer}:\n        T.cuda.thread_fence()"
        if order == "cuda"
        else f'T.ptx["{mnemonic}"](pred=lane == {consumer})'
    )
    return tvm.script.from_source(
        f"""@T.prim_func
def kernel(data: T.Buffer((1,), "int32"), output: T.Buffer((1,), "int32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    {allocation}
    if lane == 0:
        {data}[0] = 0
    T.cuda.warp_sync()
    if lane == 0:
        {data}[0] = 41
    {publication}
    {acquisition}
    if lane == 1:
        output[0] = {data}[0]
""",
        {"T": T},
    )


@pytest.mark.parametrize("space", ("global", "shared"))
def test_sc_causality_and_lane_controls(space):
    for order, publisher, consumer, expected in (
        ("sc", 0, 1, "clean"),
        ("cuda", 0, 1, "clean"),
        ("", 0, 1, "error"),
        ("acq_rel", 0, 1, "error"),
        ("sc", 2, 1, "error"),
        ("sc", 0, 2, "error"),
    ):
        kernel = _fence_case(space, order, publisher, consumer)
        inputs = dict(data=np.zeros(1, np.int32), output=np.zeros(1, np.int32))
        synccheck(kernel, inputs).require_clean()
        report = racecheck(kernel, inputs)
        _check_race_report(report, expected, space)
