from __future__ import annotations

from pathlib import Path

import numpy as np
from tvm.script import tirx as T

from tirx_harness import numsim


@T.prim_func
def tcgen_matrix_descriptor_swizzle1(output: T.Buffer((1,), "uint64")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((512,), "uint8", scope="shared", align=128)
    descriptor: T.uint64
    if lane == 0:
        T.cuda.tcgen05.encode_matrix_descriptor(
            T.address_of(descriptor),
            T.address_of(shared[0]),
            ldo=0,
            sdo=0,
            swizzle=1,
        )
        output[0] = descriptor


@T.prim_func
def tcgen_matrix_descriptor_cluster_rank(output: T.Buffer((2,), "uint64")):
    T.device_entry()
    _cluster = T.cluster_id([1])
    cta = T.cta_id_in_cluster([2])
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((512,), "uint8", scope="shared", align=128)
    descriptor: T.uint64
    if lane == 0:
        T.cuda.tcgen05.encode_matrix_descriptor(
            T.address_of(descriptor),
            T.address_of(shared[0]),
            ldo=0,
            sdo=0,
            swizzle=1,
        )
        output[cta] = descriptor


@T.prim_func
def tcgen_lifecycle_columns256(output: T.Buffer((1,), "uint32")):
    T.device_entry()
    _cta = T.cta_id([1])
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    address = T.alloc_buffer((1,), "uint32", scope="shared")
    T.ptx.tcgen05.alloc.cta_group__1.sync.aligned.shared__cta.b32(T.address_of(address[0]), 256)
    if lane == 0:
        output[0] = address[0]
    T.ptx.tcgen05.relinquish_alloc_permit.cta_group__1.sync.aligned()
    T.ptx.tcgen05.dealloc.cta_group__1.sync.aligned.b32(T.uint32(0), 256)


@T.prim_func
def tcgen_commit_uint32_predicate(output: T.Buffer((2,), "uint32")):
    T.device_entry()
    _cluster = T.cluster_id([1])
    cta = T.cta_id_in_cluster([2])
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    barrier = T.alloc_buffer((1,), "uint64", scope="shared")
    if lane == 0:
        T.ptx.mbarrier.init.shared.b64(T.address_of(barrier[0]), 1)
    T.ptx.fence.proxy.async_.shared__cta()
    T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.cluster_sync()
    if cta == 0 and lane == 0:
        T.ptx.tcgen05.commit.cta_group__2.mbarrier__arrive__one.shared__cluster.multicast__cluster.b64(
            T.address_of(barrier[0]), T.uint16(3), pred=T.uint32(1)
        )
    T.cuda.mbarrier_wait(T.address_of(barrier[0]), 0)
    if lane == 0:
        output[cta] = T.uint32(1)


def test_tcgen_matrix_descriptor_swizzle1_packs_reviewed_fields(tmp_path: Path):
    module = numsim.transpile(tcgen_matrix_descriptor_swizzle1, cache_dir=tmp_path)
    result = numsim.Engine(max_workers=1).run(
        module,
        {"output": np.zeros(1, dtype=np.uint64)},
    )

    def check() -> None:
        descriptor = int(result.outputs["output"][0])
        assert (descriptor >> 16) & 0x3FFF == 0
        assert (descriptor >> 32) & 0x3FFF == 0
        assert (descriptor >> 46) & 1 == 1
        assert descriptor >> 61 == 6

    check()


def test_tcgen_matrix_descriptor_encodes_offset_without_cluster_rank(tmp_path: Path):
    module = numsim.transpile(tcgen_matrix_descriptor_cluster_rank, cache_dir=tmp_path)
    result = numsim.Engine(max_workers=2).run(
        module,
        {"output": np.zeros(2, dtype=np.uint64)},
    )

    assert result.outputs["output"][0] != 0
    assert result.outputs["output"][0] == result.outputs["output"][1]


def test_tcgen_lifecycle_accepts_full_256_column_allocation(tmp_path: Path):
    module = numsim.transpile(tcgen_lifecycle_columns256, cache_dir=tmp_path)
    result = numsim.Engine(max_workers=1).run(
        module,
        {"output": np.full(1, np.uint32(0xFFFFFFFF), dtype=np.uint32)},
    )
    np.testing.assert_array_equal(
        result.outputs["output"],
        np.zeros(1, dtype=np.uint32),
    )


def test_tcgen_commit_accepts_explicit_uint32_predicate(tmp_path: Path):
    module = numsim.transpile(tcgen_commit_uint32_predicate, cache_dir=tmp_path)
    result = numsim.Engine(max_workers=2).run(
        module,
        {"output": np.zeros(2, dtype=np.uint32)},
    )
    np.testing.assert_array_equal(
        result.outputs["output"],
        np.ones(2, dtype=np.uint32),
    )
