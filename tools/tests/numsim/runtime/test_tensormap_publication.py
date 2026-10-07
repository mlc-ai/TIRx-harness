"""Descriptor publication must precede acquisition along actual causal edges."""

import numpy as np
import tvm
from tvm.script import tirx as T

from tirx_harness import numsim, racecheck
from tests.numsim.runtime.test_raw_tensor_map_descriptors import (
    _GDN_ACQUIRE_SOURCE,
    _GDN_RELEASE_SOURCE,
    _gdn_replace_dim_source,
)


PUBLICATION_CASES = ("replace", "bytes", "helper")


def tensor_map_publication_case(kind, ordering="before", cross_warp=False):
    acquire = "T.ptx.fence.proxy.tensormap__generic.acquire.gpu(descriptor.ptr_to([0]))"
    release = "T.ptx.fence.proxy.tensormap__generic.release.gpu()"
    update = '''T.ptx.tensormap_replace.tile.global_dim.global_.b1024.b32(
            descriptor.ptr_to([0]), 0, T.uint32(2))'''
    if kind == "bytes":
        update = '''for i in T.serial(128):
            descriptor[i] = source_map[i]'''
    elif kind == "helper":
        update = (
            "T.cuda.func_call('gdn_tensormap_replace_global_dim_0', descriptor.ptr_to([0]), "
            f"T.uint32(2), source_code={_gdn_replace_dim_source(0)!r}, return_type='void')"
        )
        acquire = (
            "T.cuda.func_call('gdn_tensormap_acquire', descriptor.ptr_to([0]), "
            f"source_code={_GDN_ACQUIRE_SOURCE!r}, return_type='void')"
        )
        release = (
            f"T.cuda.func_call('gdn_tensormap_release', source_code={_GDN_RELEASE_SOURCE!r}, "
            "return_type='void')"
        )
    sync = "T.cuda.cta_sync()" if cross_warp else "T.cuda.warp_sync()"
    kernel = tvm.script.from_source(f'''
@T.prim_func
def kernel(descriptor: T.Buffer((128,), "uint8"), source_map: T.Buffer((128,), "uint8"),
           output: T.Buffer((4,), "uint32")):
    T.device_entry()
    warp = T.warp_id([{2 if cross_warp else 1}])
    lane = T.lane_id([32])
    data = T.alloc_shared((4,), "uint32", align=128)
    barrier = T.alloc_shared((1,), "uint64")
    ready = T.alloc_local((1,), "uint32")
    if warp == 0 and lane == 0:
        T.ptx.mbarrier.init.shared.b64(barrier.ptr_to([0]), 1)
        {acquire if ordering == "stale" else "T.evaluate(0)"}
    T.ptx.fence.mbarrier_init.release.cluster()
    {sync}
    if warp == 0 and lane == 0:
        {update}
    {sync if ordering != "after" else "T.evaluate(0)"}
    if warp == {int(cross_warp)} and lane == 1:
        {release}
    {sync}
    if warp == 0 and lane == 0:
        {acquire if ordering != "stale" else "T.evaluate(0)"}
        T.ptx.mbarrier.arrive.expect_tx.shared.b64(barrier.ptr_to([0]), 16)
        T.ptx["cp.async.bulk.tensor.1d.shared::cta.global.mbarrier::complete_tx::bytes"](
            data.ptr_to([0]), descriptor.ptr_to([0]), 0, barrier.ptr_to([0]))
        ready[0] = T.uint32(0)
        while ready[0] == 0:
            T.ptx.mbarrier.try_wait.parity.shared.b64(
                ready[0], barrier.ptr_to([0]), T.uint32(0), T.uint32(1))
        for i in T.serial(4):
            output[i] = data[i]
''', {"T": T})
    source = np.arange(1, 5, dtype=np.uint32)
    metadata = dict(global_shape=(4,), global_strides=(), box_shape=(4,), element_strides=(1,))
    image = numsim.TensorMap(base=source, **metadata).numpy()
    inputs = dict(descriptor=image.copy(), source_map=image.copy(), output=np.zeros(4, np.uint32))
    expected = source.copy() if kind == "bytes" else np.array([1, 2, 0, 0], np.uint32)
    return kernel, inputs, source, metadata, expected


def test_descriptor_publication_does_not_capture_future_writes():
    for kind in PUBLICATION_CASES:
        for ordering, cross_warp in (("after", False), ("after", True), ("stale", False)):
            kernel, inputs, _source, _metadata, _expected = tensor_map_publication_case(
                kind, ordering, cross_warp
            )
            report = racecheck(kernel, inputs)
            assert report.verdict == "error", report.format()
            assert "not acquired" in report.format() or "dirty" in report.format(), report.format()
