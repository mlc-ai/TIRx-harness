"""SM107-only register predicates reuse the existing register and sparse algorithms."""

import numpy as np
import pytest
import tvm
from tvm.script import tirx as T

from tirx_harness import numsim, racecheck, synccheck


def sm107_register_predicate_kernel(preserve):
    return tvm.script.from_source(
        f"""
@T.prim_func
def kernel(output: T.Buffer((5, 32), "uint32"), selected: T.uint32):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    active = (selected & (T.uint32(1) << T.Cast("uint32", lane))) != T.uint32(0)
    src = T.alloc_local((5,), "uint32")
    dst = T.alloc_local((5,), "uint32")
    for i in T.unroll(5):
        dst[i] = 91
    if active:
        src[0] = T.uint32(0x03020401)
        src[1] = T.uint32(0x03020401)
        src[2] = T.uint32(0xDD)
        src[3] = T.uint32(0x03040304)
        src[4] = T.uint32(0)
    T.ptx["set.lt.u8x4"](dst[0], src[0], src[3], pred=active, preserve_dst={preserve})
    T.ptx["spcompress.b8.b2.sp::2:4.x1"](dst[1], dst[2], src[0], src[1], src[4],
        pred=active, preserve_dst={preserve})
    T.ptx["spdecompress.b8.b2.sp::2:4.x2"](dst[3], dst[4], src[2], src[3],
        pred=active, preserve_dst={preserve})
    for i in T.unroll(5):
        output[i, lane] = dst[i]
""",
        {"T": T},
    )


def sm107_register_predicate_expected(mask, preserve):
    expected = np.full((5, 32), 91 if preserve else 0, np.uint32)
    active = [lane for lane in range(32) if mask & (1 << lane)]
    # Byte-wise less-than; max-selection of [1,4,2,3] keeps indices 1,3;
    # decompression scatters [4,3,4,3] into those positions and zero-fills.
    expected[:, active] = np.array(
        [0x00FF00FF, 0xDD, 0x03040304, 0x03000400, 0x03000400], np.uint32
    )[:, None]
    return expected


@pytest.mark.parametrize("preserve", [False, True])
def test_sm107_register_predicates(preserve, tmp_path):
    kernel = sm107_register_predicate_kernel(preserve)
    module = numsim.transpile(kernel, cache_dir=tmp_path)
    for mask in (0, 0x80000000, 0xAAAAAAAA, 0xFFFFFFFF):
        inputs = {"output": np.zeros((5, 32), np.uint32), "selected": mask}
        for checker in (synccheck, racecheck):
            checker(kernel, inputs).require_clean()
        result = numsim.Engine().run(module, inputs)
        np.testing.assert_array_equal(
            result.outputs["output"], sm107_register_predicate_expected(mask, preserve)
        )
