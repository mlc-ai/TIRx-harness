"""Predication gates the commit's barrier address, multicast mask and arrival."""

import numpy as np
import pytest
import tvm
from tvm.script import tirx as T

from tirx_harness import numsim, racecheck, synccheck


def commit_kernel(restricted, multicast):
    restrict = ".sync_restrict::shared::read::mma::a" if restricted else ""
    suffix = f".multicast::cluster{multicast}" if multicast is not None else ""
    mask_dtype = "uint32" if multicast == "::32b" else "uint16"
    extra = ", targets[0]" if multicast is not None else ""
    return tvm.script.from_source(
        f'''
@T.prim_func
def kernel(output: T.Buffer((32,), "uint32"), enabled: T.int32, selected: T.int32):
    T.device_entry()
    _cluster = T.cluster_id([1])
    _cta = T.cta_id_in_cluster([1])
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    barrier = T.alloc_shared((1,), "uint64")
    pointer = T.alloc_local((1,), "uint32")
    targets = T.alloc_local((1,), "{mask_dtype}")
    if enabled != 0 and lane == selected:
        T.ptx.mbarrier.init.shared.b64(barrier.ptr_to([0]), 1)
        pointer[0] = T.cuda.cvta_generic_to_shared(barrier.ptr_to([0]))
        targets[0] = T.Cast("{mask_dtype}", 1)
    T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.cta_sync()
    T.ptx["tcgen05.commit.cta_group::1.mbarrier::arrive::one{restrict}.shared::cluster{suffix}.b64"](
        pointer[0]{extra}, pred=enabled != 0 and lane == selected)
    if enabled != 0:
        T.cuda.mbarrier_wait(barrier.ptr_to([0]), 0)
        output[lane] = T.Cast("uint32", lane + 42)
''',
        {"T": T},
    )


def commit_inputs(enabled, selected):
    inputs = {"output": np.full(32, 77, np.uint32), "enabled": enabled, "selected": selected}
    expected = np.arange(32, dtype=np.uint32) + 42 if enabled else inputs["output"].copy()
    return inputs, expected


@pytest.mark.parametrize("restricted", [False, True])
def test_tcgen_commit_predicates_gate_all_operands(restricted, tmp_path):
    for multicast in (None, "", "::16b", "::32b"):
        kernel = commit_kernel(restricted, multicast)
        module = numsim.transpile(kernel, cache_dir=tmp_path)
        for enabled, selected in ((0, 0), (1, 0), (1, 31)):
            inputs, expected = commit_inputs(enabled, selected)
            for checker in (synccheck, racecheck):
                checker(kernel, inputs).require_clean()
            result = numsim.Engine().run(module, inputs)
            np.testing.assert_array_equal(result.outputs["output"], expected)
