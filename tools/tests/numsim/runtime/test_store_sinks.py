"""A vector store's sink components neither write bytes nor create conflicts."""

import numpy as np
import tvm
from tvm.script import tirx as T

from tirx_harness import numsim, racecheck, synccheck


def sink_release_kernel(bits, release=True, sinks=True, space="global"):
    width = 2 if space == "shared" else 256 // bits
    scope = "cta" if space == "shared" else "gpu"
    owner = "shared_flag" if space == "shared" else "flag"
    setup = (
        f'shared_flag = T.alloc_shared(({width},), "uint{bits}", align=32)\n'
        f"    if warp == 0 and lane < {width}:\n"
        f"        shared_flag[lane] = T.uint{bits}(0)\n"
        "    T.cuda.cta_sync()"
        if space == "shared"
        else ""
    )
    values = ", ".join(
        f"T.uint{bits}(1)" if i < 2 or not sinks else "T.ptx.SINK" for i in range(width)
    )
    return tvm.script.from_source(
        f"""
@T.prim_func
def kernel(flag: T.Buffer(({width},), "uint{bits}"), data: T.Buffer((1,), "uint32"),
           output: T.Buffer((1,), "uint32")):
    T.device_entry()
    warp = T.warp_id([2])
    lane = T.lane_id([32])
    ready = T.alloc_local((1,), "uint{bits}")
    {setup}
    if warp == 0 and lane == 0:
        data[0] = T.uint32(79)
        T.ptx["st.{"release" if release else "relaxed"}.{scope}.{space}.v{width}.u{bits}"](
            {owner}.ptr_to([0]), {values})
    elif warp == 1 and lane == 0:
        ready[0] = T.uint{bits}(0)
        while ready[0] == 0:
            T.ptx["ld.acquire.{scope}.{space}.u{bits}"](ready[0], {owner}.ptr_to([1]))
        output[0] = data[0]
""",
        {"T": T},
    )


def release_inputs(bits, space="global"):
    width = 2 if space == "shared" else 256 // bits
    return dict(
        flag=np.zeros(width, f"uint{bits}"),
        data=np.zeros(1, np.uint32),
        output=np.zeros(1, np.uint32),
    )


def test_store_sinks_retain_per_element_release(tmp_path):
    for bits in (32, 64):
        for release, sinks, space in (
            (True, False, "global"),
            (True, True, "global"),
            (False, True, "global"),
            (True, False, "shared"),
        ):
            kernel = sink_release_kernel(bits, release, sinks, space)
            inputs = release_inputs(bits, space)
            synccheck(kernel, inputs).require_clean()
            report = racecheck(kernel, inputs)
            if release:
                report.require_clean()
                result = numsim.Engine().run(numsim.transpile(kernel, cache_dir=tmp_path), inputs)
                np.testing.assert_array_equal(result.outputs["output"], [79])
            else:
                assert report.verdict == "error", report.format()
                assert any(
                    finding.details["access_pair"] in {"read_write", "write_read"} for finding in report.findings
                ), report.format()
