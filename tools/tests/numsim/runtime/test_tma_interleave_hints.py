"""Shared-layout coverage gaps must not reject descriptor-only cache hints."""

import numpy as np
import pytest
import tvm
from tvm.script import tirx as T

from tests.numsim.support.execution import assert_rejected, run_checked
from tirx_harness import numsim


def interleave_hint_case(operation="prefetch", *, raw=False, swizzle="32B"):
    descriptor_type = 'T.Buffer((128,), "uint8")' if raw else "T.TensorMap()"
    pointer = "descriptor.ptr_to([0])" if raw else "T.address_of(descriptor)"
    if operation == "prefetch":
        body = f"""T.ptx["cp.async.bulk.prefetch.tensor.3d.L2.global.tile"](
            {pointer}, 0, 0, 0, pred=lane == 0)"""
    elif operation == "load":
        body = f"""barrier = T.alloc_shared((1,), "uint64")
    if lane == 0:
        T.ptx.mbarrier.init.shared.b64(barrier.ptr_to([0]), 1)
    T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.cta_sync()
    if lane == 0 and enabled != 0:
        T.ptx.mbarrier.arrive.expect_tx.shared.b64(barrier.ptr_to([0]), 1024)
    T.ptx["cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes"](
        shared.ptr_to([0]), {pointer}, 0, 0, 0, barrier.ptr_to([0]), pred=(lane == 0 and enabled != 0))
    if lane == 0 and enabled != 0:
        T.cuda.mbarrier_wait(barrier.ptr_to([0]), 0)"""
    elif operation in {"store", "reduce"}:
        opcode = (
            "cp.async.bulk.tensor.3d.global.shared::cta.tile.bulk_group"
            if operation == "store"
            else "cp.reduce.async.bulk.tensor.3d.global.shared::cta.add.bulk_group"
        )
        body = f'''T.ptx["{opcode}"](
        {pointer}, 0, 0, 0, shared.ptr_to([0]), pred=(lane == 0 and enabled != 0))
    if lane == 0:
        T.ptx.cp.async_.bulk.commit_group()
        T.ptx.cp.async_.bulk.wait_group(0)'''
    else:
        raise ValueError(operation)
    if raw:
        body = f"""if lane == 0:
        T.ptx.fence.proxy.tensormap__generic.acquire.gpu({pointer})
    {body}"""
    kernel = tvm.script.from_source(
        f"""@T.prim_func
def kernel(descriptor: {descriptor_type}, enabled: T.int32, output: T.Buffer((32,), "uint32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_shared((256,), "uint32", align=1024)
    for row in T.serial(8):
        shared[row * 32 + lane] = T.uint32(0)
    T.ptx.fence.proxy.async_.shared__cta()
    T.cuda.cta_sync()
    {body}
    output[lane] = T.uint32(0xabc000) + lane
""",
        {"T": T},
    )
    base = np.arange(1024, dtype=np.uint32)
    metadata = dict(
        global_shape=(8, 4, 8),
        global_strides=(128, 512),
        box_shape=(8, 4, 8),
        element_strides=(1, 1, 1),
        interleave="16B",
        swizzle=swizzle,
    )
    return kernel, {"enabled": 0, "output": np.zeros(32, np.uint32)}, base, metadata


@pytest.mark.parametrize("raw,swizzle", [(False, "32B"), (True, "128B_ATOM_32B")])
def test_swizzled_interleave_prefetch_preserves_memory(raw, swizzle):
    kernel, inputs, base, metadata = interleave_hint_case(raw=raw, swizzle=swizzle)
    inputs["descriptor"] = numsim.TensorMap(base, **metadata).numpy()
    before = base.copy()
    result = run_checked(kernel, inputs, outputs=("output",))
    np.testing.assert_array_equal(
        result.outputs["output"], np.arange(32, dtype=np.uint32) + 0xABC000
    )
    np.testing.assert_array_equal(base, before)


@pytest.mark.parametrize("operation", ["load", "store", "reduce"])
def test_swizzled_interleave_transfer_rejects_only_when_issued(operation):
    kernel, inputs, base, metadata = interleave_hint_case(operation)
    inputs["descriptor"] = numsim.TensorMap(base, **metadata).numpy()
    result = run_checked(kernel, inputs, outputs=("output",))
    np.testing.assert_array_equal(
        result.outputs["output"], np.arange(32, dtype=np.uint32) + 0xABC000
    )
    inputs["enabled"] = 1
    reports = assert_rejected(
        kernel,
        inputs,
        "tma_swizzled_16b_interleave_unmodeled",
        verdict="incomplete",
    )
    for report in reports:
        assert any(finding.details.get("operation") for finding in report.findings), report.format()


def test_swizzled_interleave_still_rejects_malformed_geometry():
    _, _, base, metadata = interleave_hint_case()
    with pytest.raises(ValueError, match="96B swizzle does not support interleave"):
        numsim.TensorMap(base, **{**metadata, "swizzle": "96B"}).numpy()
