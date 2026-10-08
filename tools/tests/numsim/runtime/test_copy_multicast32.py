"""Exercise high multicast bits, not just 32-bit spellings of low masks."""

import numpy as np
import pytest
import tvm

from tirx_harness import numsim, racecheck, synccheck
from tests.numsim.support.execution import run_checked
from tests.numsim.runtime.test_raw_tma_codegen import _tensor_map


def high_bit_multicast_kernel(form, *, ctas=20, mask=None):
    mask = (1 << (ctas - 1)) | 1 if mask is None else mask
    transfer = form != "commit"
    parameter = (
        "input_map: T.TensorMap()"
        if form in {"tensor", "im2col"}
        else 'source: T.Buffer((4,), "float32")'
    )
    instruction = {
        "bulk": f"""T.ptx["cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster::32b"](
            shared.ptr_to([0]), source.ptr_to([0]), T.uint32(16), barrier.ptr_to([0]), T.uint32({mask}))""",
        "tensor": f"""T.ptx["cp.async.bulk.tensor.1d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster::32b.cta_group::1"](
            shared.ptr_to([0]), T.address_of(input_map), 0, barrier.ptr_to([0]), T.uint32({mask}))""",
        "im2col": f"""T.ptx["cp.async.bulk.tensor.3d.shared::cluster.global.im2col.mbarrier::complete_tx::bytes.multicast::cluster::32b.cta_group::1"](
            shared.ptr_to([0]), T.address_of(input_map), 0, 0, 0, barrier.ptr_to([0]), T.uint16(0), T.uint32({mask}))""",
        "commit": f"""T.ptx["tcgen05.commit.cta_group::1.mbarrier::arrive::one.multicast::cluster::32b.b64"](
            T.reinterpret("handle", barrier.ptr_to([0])), T.uint32({mask}))""",
    }[form]
    return tvm.script.from_source(
        f"""
@T.prim_func
def multicast32({parameter}, output: T.Buffer(({ctas}, 4), "float32")):
    T.device_entry()
    _cluster = T.cluster_id([1])
    cta = T.cta_id_in_cluster([{ctas}])
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((4,), "float32", scope="shared")
    barrier = T.alloc_buffer((1,), "uint64", scope="shared")
    if lane < 4 and cta != 0 and cta != {ctas - 1}:
        shared[lane] = T.float32(-7)
    if lane == 0:
        T.ptx.mbarrier.init.shared.b64(barrier.ptr_to([0]), 1)
    T.ptx.fence.proxy.async_.shared__cluster()
    T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.cluster_sync()
    if lane == 0 and (cta == 0 or cta == {ctas - 1}):
        {"T.ptx.mbarrier.arrive.expect_tx.shared.b64(barrier.ptr_to([0]), 16)" if transfer else "T.evaluate(0)"}
    if cta == 0 and lane == 0:
        {instruction}
    if lane == 0 and (cta == 0 or cta == {ctas - 1}):
        T.cuda.mbarrier_wait(barrier.ptr_to([0]), 0)
    T.cuda.cta_sync()
    if lane < 4:
        output[cta, lane] = {"shared[lane]" if transfer else "T.Cast('float32', (cta == 0) or (cta == " + str(ctas - 1) + "))"}
    T.cuda.cluster_sync()
"""
    )


def multicast_inputs(form, ctas=20):
    source = np.arange(4, dtype=np.float32) + np.float32(0.25)
    if form == "im2col":
        inputs = {
            "input_map": numsim.TensorMap(
                source, (4, 1, 1), (16, 16), (4, 1), (1, 1, 1), im2col=numsim.Im2col((0,), (0,))
            ).numpy()
        }
    elif form == "tensor":
        tensor_map, _ = _tensor_map(source, global_shape=(4,), global_strides=(), box_shape=(4,))
        inputs = {"input_map": tensor_map}
    else:
        inputs = {"source": source}
    inputs["output"] = np.zeros((ctas, 4), np.float32)
    expected = np.full((ctas, 4), -7 if form != "commit" else 0, np.float32)
    expected[[0, ctas - 1]] = source if form != "commit" else 1
    return inputs, expected


@pytest.mark.parametrize("form", ["bulk", "tensor", "commit", "im2col"])
def test_multicast_high_bit_targets_and_bounds(form, tmp_path):
    kernel = high_bit_multicast_kernel(form)
    inputs, expected = multicast_inputs(form)
    result = run_checked(kernel, inputs, cache_dir=tmp_path)
    np.testing.assert_array_equal(result.outputs["output"], expected)

    invalid = high_bit_multicast_kernel(form, mask=1 << 20)
    for checker in (synccheck, racecheck):
        report = checker(invalid, inputs)
        assert report.verdict == "error"
        assert "outside cluster" in str(report.to_dict())
