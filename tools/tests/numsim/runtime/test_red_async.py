"""Remote atomic reduction, byte completion, and ordinary consumers."""

import numpy as np
import pytest
import tvm
from tvm.script import tirx as T

from tests.numsim.support.execution import run_checked
from tirx_harness import racecheck


def reduction_kernel(op, ptx_type, *, bulk=False, wait=True, lanes=1):
    dtype = {"u32": "uint32", "s32": "int32", "u64": "uint64", "b32": "uint32"}[ptx_type]
    byte_len = np.dtype(dtype).itemsize
    elements = 16 // byte_len if bulk else 1
    instruction = (
        f'T.ptx["cp.reduce.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes.{op}.{ptx_type}"](remote_destination[0], source.ptr_to([0]), T.uint32(16), remote_barrier[0])'
        if bulk
        else f'T.ptx["red_async.relaxed.cluster.shared::cluster.mbarrier::complete_tx::bytes.{op}.{ptx_type}"](remote_destination[0], T.cast(3, "{dtype}"), remote_barrier[0])'
    )
    return tvm.script.from_source(
        f'''
@T.prim_func
def kernel(out: T.Buffer(({elements},), "{dtype}")):
    T.device_entry()
    _cluster = T.cluster_id([1])
    cta = T.cta_id_in_cluster([2])
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    destination = T.alloc_buffer(({elements},), "{dtype}", scope="shared", align=16)
    source = T.alloc_buffer(({elements},), "{dtype}", scope="shared", align=16)
    barrier = T.alloc_buffer((1,), "uint64", scope="shared")
    if lane == 0:
        for i in T.unroll({elements}):
            destination[i] = T.cast(7, "{dtype}")
            source[i] = T.cast(3, "{dtype}")
        T.ptx.mbarrier.init.shared.b64(barrier.ptr_to([0]), {lanes})
    if {bulk}:
        T.ptx.fence.proxy.async_.shared__cta()
    T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.cluster_sync()
    if (cta == 0) and (lane < {lanes}):
        remote_barrier = T.alloc_local((1,), "uint32")
        remote_destination = T.alloc_local((1,), "uint32")
        T.ptx.mapa.shared__cluster.u32(remote_barrier[0],
            T.cuda.cvta_generic_to_shared(barrier.ptr_to([0])), T.uint32(1))
        T.ptx.mapa.shared__cluster.u32(remote_destination[0],
            T.cuda.cvta_generic_to_shared(destination.ptr_to([0])), T.uint32(1))
        T.ptx.mbarrier.arrive.expect_tx.shared__cluster.b64(remote_barrier[0], T.uint32({elements * byte_len}), pred=True)
        {instruction}
    if (cta == 1) and (lane == 0):
        if {wait}:
            T.cuda.mbarrier_wait(barrier.ptr_to([0]), 0)
        for i in T.unroll({elements}):
            out[i] = destination[i]
    T.cuda.cluster_sync()
''',
        {"T": T},
    )


@pytest.mark.parametrize(
    "op,ptx_type,expected",
    [
        ("add", "u32", 10),
        ("add", "s32", 10),
        ("add", "u64", 10),
        ("min", "u32", 3),
        ("min", "s32", 3),
        ("max", "u32", 7),
        ("max", "s32", 7),
        ("and", "b32", 3),
        ("or", "b32", 7),
        ("xor", "b32", 4),
        ("inc", "u32", 0),
        ("dec", "u32", 3),
    ],
)
def test_red_async(op, ptx_type, expected, tmp_path):
    dtype = {"u32": "uint32", "s32": "int32", "u64": "uint64", "b32": "uint32"}[ptx_type]
    args = {"out": np.zeros(1, dtype=dtype)}
    kernel = reduction_kernel(op, ptx_type)
    result = run_checked(kernel, args, cache_dir=tmp_path)
    np.testing.assert_array_equal(result.outputs["out"], [expected])


@pytest.mark.parametrize(
    "bulk,ptx_type,op,expected",
    [
        (False, "u32", "add", 19),
        (True, "u32", "add", 10),
        (True, "s32", "min", 3),
        (True, "b32", "xor", 4),
        (True, "u64", "add", 10),
    ],
)
def test_shared_async_reduction_completion(bulk, ptx_type, op, expected, tmp_path):
    dtype = {"u32": "uint32", "s32": "int32", "u64": "uint64", "b32": "uint32"}[ptx_type]
    count = 16 // np.dtype(dtype).itemsize if bulk else 1
    args = {"out": np.zeros(count, dtype=dtype)}
    kernel = reduction_kernel(op, ptx_type, bulk=bulk, lanes=1 if bulk else 4)
    result = run_checked(kernel, args, cache_dir=tmp_path)
    np.testing.assert_array_equal(result.outputs["out"], np.full(count, expected, dtype=dtype))
    report = racecheck(reduction_kernel(op, ptx_type, bulk=bulk, wait=False), args)
    assert report.verdict == "error"
    assert any(
        f.status == "error" and f.details["access_pair"] in {"write_read", "read_write"} for f in report.findings
    )
