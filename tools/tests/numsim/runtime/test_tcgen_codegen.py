from dataclasses import dataclass

import numpy as np
import pytest

from tirx_harness import numsim
from tests.numsim.support.manifest import call_op_names, emitted_module, emitted_calls
from tests.numsim.support.kernels import (
    tcgen_commit_mbarrier,
    tcgen_commit_runtime_multicast,
    tcgen_lifecycle_single_cta,
    tcgen_lifecycle_two_cta,
)
from tirx_harness.numsim.transpiler.frontend import analyze
from tirx_harness.numsim.transpiler.ptx_dialect import DecodedPtxCall, decode_ptx_call
from tvm.script import tirx as T
from tvm.tirx.lang.pipeline import TCGen05Bar


@T.prim_func
def tcgen_control_calls():
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((2,), "uint64", scope="shared")
    address = shared.view("uint32")
    T.ptx.tcgen05.alloc.cta_group__1.sync.aligned.shared__cta.b32(T.address_of(address[0]), 128)
    if lane == 0:
        T.ptx.tcgen05.commit.cta_group__1.mbarrier__arrive__one.shared__cluster.b64(
            T.address_of(shared[1])
        )
        T.ptx.tcgen05.commit.cta_group__2.mbarrier__arrive__one.shared__cluster.multicast__cluster.b64(
            T.address_of(shared[1]), T.uint16(3), pred=T.uint32(1)
        )
    T.ptx.tcgen05.wait__ld.sync.aligned()
    T.ptx.tcgen05.wait__st.sync.aligned()
    T.ptx.tcgen05.fence__before_thread_sync()
    T.ptx.tcgen05.fence__after_thread_sync()
    T.ptx.setmaxnreg.inc.sync.aligned.u32(128)
    T.ptx.tcgen05.relinquish_alloc_permit.cta_group__1.sync.aligned()
    T.ptx.tcgen05.dealloc.cta_group__1.sync.aligned.b32(T.uint32(0), 128)


@T.prim_func
def tcgen_commit_raw_shared_offset(barrier_index: T.int32, output: T.Buffer((1,), "uint32")):
    T.device_entry()
    _cta = T.cta_id([1])
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    barriers = T.alloc_buffer((2,), "uint64", scope="shared")
    barrier_base: T.uint32 = T.cuda.cvta_generic_to_shared(barriers.ptr_to([0]))
    if lane == 0:
        T.ptx.mbarrier.init.shared.b64(T.address_of(barriers[barrier_index]), 1)
    T.ptx.fence.proxy.async_.shared__cta()
    T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.cta_sync()
    if lane == 0:
        T.ptx.tcgen05.commit.cta_group__1.mbarrier__arrive__one.shared__cluster.b64(
            barrier_base + T.cast(barrier_index * 8, "uint32")
        )
    T.cuda.mbarrier_wait(T.address_of(barriers[barrier_index]), 0)
    if lane == 0:
        output[0] = T.uint32(1)


@T.prim_func
def tcgen_alloc_raw_shared_offset(output: T.Buffer((1,), "uint32")):
    T.device_entry()
    _cta = T.cta_id([1])
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    addresses = T.alloc_buffer((2,), "uint32", scope="shared")
    shared_base: T.uint32 = T.cuda.cvta_generic_to_shared(addresses.ptr_to([0]))
    destination = shared_base + T.uint32(4)
    T.ptx.tcgen05.alloc.cta_group__1.sync.aligned.shared__cta.b32(destination, 32)
    T.cuda.warp_sync()
    if lane == 0:
        output[0] = addresses[1]
    T.ptx.tcgen05.dealloc.cta_group__1.sync.aligned.b32(addresses[1], 32)
    T.ptx.tcgen05.relinquish_alloc_permit.cta_group__1.sync.aligned()


@T.prim_func
def tcgen_fence_noop_forms(output: T.Buffer((32,), "uint32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    T.ptx.tcgen05.fence__before_thread_sync()
    T.cuda.warp_sync()
    T.ptx.tcgen05.fence__after_thread_sync()
    output[lane] = T.cast(lane + 1, "uint32")


@T.prim_func
def tcgen_runtime_uniform_dealloc(output: T.Buffer((1,), "uint32")):
    T.device_entry()
    _cta = T.cta_id([1])
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    address = T.alloc_buffer((1,), "uint32", scope="shared")
    runtime_address = T.alloc_local((1,), "uint32")
    T.ptx.tcgen05.alloc.cta_group__1.sync.aligned.shared__cta.b32(T.address_of(address[0]), 64)
    runtime_address[0] = address[0]
    if lane == 0:
        output[0] = runtime_address[0]
    T.ptx.tcgen05.relinquish_alloc_permit.cta_group__1.sync.aligned()
    T.ptx.tcgen05.dealloc.cta_group__1.sync.aligned.b32(runtime_address[0], 64)


@T.prim_func
def tcgen_runtime_divergent_dealloc(output: T.Buffer((1,), "uint32")):
    T.device_entry()
    _cta = T.cta_id([1])
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    runtime_address = T.alloc_local((1,), "uint32")
    runtime_address[0] = lane
    T.ptx.tcgen05.dealloc.cta_group__1.sync.aligned.b32(runtime_address[0], 64)
    if lane == 0:
        output[0] = T.uint32(1)


@T.prim_func
def tcgen_alloc_non_power_of_two_columns():
    T.device_entry()
    _warp = T.warp_id([1])
    address = T.alloc_buffer((1,), "uint32", scope="shared")
    T.ptx.tcgen05.alloc.cta_group__1.sync.aligned.shared__cta.b32(T.address_of(address[0]), 96)


@T.prim_func
def tmem_pool_non_power_of_two_columns(output: T.Buffer((1,), "int32")):
    T.device_entry()
    T.cta_id([1])
    T.warp_id([1])
    thread_id = T.thread_id([32])

    pool = T.SMEMPool()
    tmem_address = pool.alloc((1,), "uint32", align=4)
    tmem_pool = T.TMEMPool(
        pool,
        total_cols=160,
        cta_group=1,
        tmem_addr=tmem_address,
    )
    tmem_pool.alloc((128, 160), "float32")
    pool.commit()
    tmem_pool.commit()
    tmem_pool.dealloc()
    if thread_id == 0:
        output[0] = 1


@T.prim_func
def tcgen_dealloc_non_power_of_two_columns():
    T.device_entry()
    T.warp_id([1])
    T.lane_id([32])
    T.ptx.tcgen05.dealloc.cta_group__1.sync.aligned.b32(T.uint32(0), 160)


@T.prim_func
def tcgen_commit_static_single_bit_mask_is_unicast(output: T.Buffer((1,), "uint32")):
    T.device_entry()
    _cluster = T.cluster_id([1])
    cta = T.cta_id_in_cluster([2])
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    pool = T.SMEMPool()
    barrier = TCGen05Bar(pool, 1)
    pool.commit()
    barrier.init(1)
    T.ptx.fence.proxy.async_.shared__cluster()
    T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.cluster_sync()
    if (cta == 0) and (lane == 0):
        barrier.arrive(0, cta_group=1, cta_mask=2)
        barrier.wait(0, 0)
        output[0] = T.uint32(1)


@T.prim_func
def tcgen_commit_dynamic_single_bit_mask_is_multicast(
    cta_mask: T.int32, output: T.Buffer((1,), "uint32")
):
    T.device_entry()
    _cluster = T.cluster_id([1])
    cta = T.cta_id_in_cluster([2])
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    barrier = T.alloc_buffer((1,), "uint64", scope="shared")
    if lane == 0:
        T.ptx.mbarrier.init.shared.b64(T.address_of(barrier[0]), 1)
    T.ptx.fence.proxy.async_.shared__cluster()
    T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.cluster_sync()
    if (cta == 0) and (lane == 0):
        T.ptx.tcgen05.commit.cta_group__1.mbarrier__arrive__one.shared__cluster.multicast__cluster.b64(
            T.address_of(barrier[0]), T.uint16(cta_mask)
        )
    if (cta == 1) and (lane == 0):
        T.cuda.mbarrier_wait(T.address_of(barrier[0]), 0)
        output[0] = T.uint32(1)


@T.prim_func
def tcgen_commit_dynamic_uint16_mask():
    T.device_entry()
    cta, _ = T.cta_id_in_cluster([4, 1])
    lane = T.thread_id([32])
    barrier = T.alloc_buffer((1,), "uint64", scope="shared")
    mask: T.uint16 = T.if_then_else(cta < 2, T.uint16(3), T.uint16(12))
    if lane == 0:
        T.ptx.tcgen05.commit.cta_group__2.mbarrier__arrive__one.shared__cluster.multicast__cluster.b64(
            T.address_of(barrier[0]), T.uint16(mask)
        )
        T.ptx.tcgen05.commit.cta_group__2.mbarrier__arrive__one.shared__cluster.multicast__cluster.b64(
            T.address_of(barrier[0]), T.uint16(mask), pred=T.uint32(1)
        )


@dataclass(frozen=True)
class _ResolvedSource:
    op_name: str
    payload: DecodedPtxCall


def _resolved_sources(func, op_names):
    kernel = analyze(func).kernels[0]
    return [
        _ResolvedSource(str(getattr(source.node.op, "name", "")), decode_ptx_call(source.node))
        for source in kernel.source_map
        if source.kind == "Call" and str(getattr(source.node.op, "name", "")) in op_names
    ]


_TCGEN_CONTROL_OPS = {
    "tirx.ptx.tcgen05_alloc",
    "tirx.ptx.tcgen05_alloc_exclusive",
    "tirx.ptx.tcgen05_commit",
    "tirx.ptx.tcgen05_commit_multicast",
    "tirx.ptx.tcgen05_commit_multicast_width",
    "tirx.ptx.tcgen05_dealloc",
    "tirx.ptx.tcgen05_dealloc_exclusive",
    "tirx.ptx.tcgen05_fence",
    "tirx.ptx.tcgen05_relinquish_alloc_permit",
    "tirx.ptx.tcgen05_wait",
}


def _classified_calls():
    return _resolved_sources(tcgen_control_calls, _TCGEN_CONTROL_OPS)


def test_tcgen_control_registry_covers_exact_source_signatures():
    calls = _classified_calls()
    assert [call.op_name for call in calls] == [
        "tirx.ptx.tcgen05_alloc",
        "tirx.ptx.tcgen05_commit",
        "tirx.ptx.tcgen05_commit_multicast",
        "tirx.ptx.tcgen05_wait",
        "tirx.ptx.tcgen05_wait",
        "tirx.ptx.tcgen05_fence",
        "tirx.ptx.tcgen05_fence",
        "tirx.ptx.tcgen05_relinquish_alloc_permit",
        "tirx.ptx.tcgen05_dealloc",
    ]
    assert [
        call.payload.modifier("action")
        for call in calls
        if call.op_name in {"tirx.ptx.tcgen05_wait", "tirx.ptx.tcgen05_fence"}
    ] == [
        "wait::ld",
        "wait::st",
        "fence::before_thread_sync",
        "fence::after_thread_sync",
    ]


def test_tcgen_fence_forms_are_numerical_noops(tmp_path):
    module = numsim.transpile(tcgen_fence_noop_forms, cache_dir=tmp_path)
    result = numsim.Engine().run(module, {"output": np.zeros(32, dtype=np.uint32)})

    np.testing.assert_array_equal(result.outputs["output"], np.arange(1, 33, dtype=np.uint32))
    assert "tirx.ptx.tcgen05_fence" in call_op_names(module.spec.kernels[0])
    assert "tirx.ptx.tcgen05_fence" not in module.rust_source


def test_frontend_records_dynamic_tmem_lifecycle_from_tcgen_alloc():
    assert analyze(tcgen_control_calls).kernels[0].uses_dynamic_tmem_lifecycle


def test_tcgen_registry_defers_invalid_columns_to_engine():
    # Columns and cta_group are runtime operands, so an out-of-contract column
    # count resolves to the one `alloc` spelling and is rejected by the engine.
    (alloc,) = emitted_calls(tcgen_alloc_non_power_of_two_columns, "tirx.ptx.tcgen05_alloc")
    assert alloc.head == "v2::tcgen05::alloc::<v2::tcgen05::variant::Alloc>"


def test_tmem_pool_non_power_of_two_columns_fail_at_engine(tmp_path):
    module = numsim.transpile(tmem_pool_non_power_of_two_columns, cache_dir=tmp_path)

    with pytest.raises(
        numsim.NumSimExecutionError,
        match=r"tcgen_invalid_columns at tcgen05\.alloc.*got 160",
    ):
        numsim.Engine().run(module, {"output": np.zeros(1, dtype=np.int32)})


def test_tcgen_dealloc_non_power_of_two_columns_fail_at_engine(tmp_path):
    module = numsim.transpile(tcgen_dealloc_non_power_of_two_columns, cache_dir=tmp_path)

    with pytest.raises(
        numsim.NumSimExecutionError,
        match=r"tcgen_invalid_columns at tcgen05\.dealloc.*got 160",
    ):
        numsim.Engine().run(module, {})


def test_tcgen_registry_accepts_runtime_cta_mask():
    commits = _resolved_sources(
        tcgen_commit_runtime_multicast, {"tirx.ptx.tcgen05_commit_multicast"}
    )
    assert len(commits) == 1
    mask = commits[0].payload.scalar_operand("mask")
    assert str(mask.ty.dtype) == "uint16"
    assert type(mask).__name__ != "IntImm"


def test_tcgen_registry_accepts_dynamic_uint16_cta_mask(tmp_path):
    commits = _resolved_sources(
        tcgen_commit_dynamic_uint16_mask, {"tirx.ptx.tcgen05_commit_multicast"}
    )
    assert [str(commit.payload.scalar_operand("mask").ty.dtype) for commit in commits] == [
        "uint16",
        "uint16",
    ]
    # Only the second commit carries a predicate, so only it opens a mask region.
    body = emitted_module(tcgen_commit_dynamic_uint16_mask)
    assert body.count("v2::tcgen05::commit::<") == 2
    assert body.count("let tcgen_control_mask_") == 1
    assert (
        body.index("v2::tcgen05::commit::<")
        < body.index("let tcgen_control_mask_")
        < body.rindex("v2::tcgen05::commit::<")
    )

    module = numsim.transpile(tcgen_commit_dynamic_uint16_mask, cache_dir=tmp_path)
    assert "tirx.ptx.tcgen05_commit_multicast" in call_op_names(module.spec.kernels[0])


def test_tcgen_single_cta_lifecycle_writes_canonical_base(tmp_path):
    output = np.full(1, np.uint32(0xFFFFFFFF), dtype=np.uint32)
    module = numsim.transpile(tcgen_lifecycle_single_cta, cache_dir=tmp_path)
    result = numsim.Engine().run(module, {"output": output})

    np.testing.assert_array_equal(result.outputs["output"], np.zeros(1, dtype=np.uint32))


def test_tcgen_two_cta_lifecycle_rendezvous(tmp_path):
    output = np.full(2, np.uint32(0xFFFFFFFF), dtype=np.uint32)
    module = numsim.transpile(tcgen_lifecycle_two_cta, cache_dir=tmp_path)
    result = numsim.Engine().run(module, {"output": output})

    np.testing.assert_array_equal(result.outputs["output"], np.zeros(2, dtype=np.uint32))
    assert result.stats["completed_task_count"] == 2


def test_tcgen_dealloc_runtime_proves_lane_local_address_uniform(tmp_path):
    output = np.full(1, np.uint32(0xFFFFFFFF), dtype=np.uint32)
    module = numsim.transpile(tcgen_runtime_uniform_dealloc, cache_dir=tmp_path)
    result = numsim.Engine().run(module, {"output": output})

    np.testing.assert_array_equal(result.outputs["output"], np.zeros(1, dtype=np.uint32))


def test_tcgen_dealloc_runtime_rejects_lane_disagreement(tmp_path):
    output = np.zeros(1, dtype=np.uint32)
    module = numsim.transpile(tcgen_runtime_divergent_dealloc, cache_dir=tmp_path)
    with pytest.raises(numsim.NumSimExecutionError, match="must agree across active lanes"):
        numsim.Engine().run(module, {"output": output})


def test_tcgen_commit_arrives_on_physical_mbarrier(tmp_path):
    output = np.zeros(1, dtype=np.uint32)
    module = numsim.transpile(tcgen_commit_mbarrier, cache_dir=tmp_path)
    result = numsim.Engine().run(module, {"output": output})

    np.testing.assert_array_equal(result.outputs["output"], np.ones(1, dtype=np.uint32))


def test_tcgen_commit_accepts_bound_raw_shared_barrier_offset(tmp_path):
    output = np.zeros(1, dtype=np.uint32)
    module = numsim.transpile(tcgen_commit_raw_shared_offset, cache_dir=tmp_path)
    result = numsim.Engine().run(
        module,
        {"barrier_index": 1, "output": output},
    )

    np.testing.assert_array_equal(result.outputs["output"], np.ones(1, dtype=np.uint32))


def test_tcgen_alloc_accepts_bound_raw_shared_destination_offset(tmp_path):
    output = np.full(1, np.uint32(0xFFFFFFFF), dtype=np.uint32)
    module = numsim.transpile(tcgen_alloc_raw_shared_offset, cache_dir=tmp_path)
    result = numsim.Engine().run(module, {"output": output})

    np.testing.assert_array_equal(result.outputs["output"], np.zeros(1, dtype=np.uint32))


def test_tcgen_commit_runtime_mask_multicasts_same_barrier_offset(tmp_path):
    output = np.zeros(2, dtype=np.uint32)
    module = numsim.transpile(tcgen_commit_runtime_multicast, cache_dir=tmp_path)
    result = numsim.Engine().run(module, {"output": output})

    np.testing.assert_array_equal(result.outputs["output"], np.ones(2, dtype=np.uint32))


def test_tcgen_commit_static_single_bit_mask_is_unicast_from_issuer(tmp_path):
    output = np.zeros(1, dtype=np.uint32)
    module = numsim.transpile(tcgen_commit_static_single_bit_mask_is_unicast, cache_dir=tmp_path)

    result = numsim.Engine().run(module, {"output": output})

    np.testing.assert_array_equal(result.outputs["output"], np.ones(1, dtype=np.uint32))


def test_tcgen_commit_dynamic_single_bit_mask_retains_multicast_semantics(tmp_path):
    output = np.zeros(1, dtype=np.uint32)
    module = numsim.transpile(tcgen_commit_dynamic_single_bit_mask_is_multicast, cache_dir=tmp_path)

    result = numsim.Engine().run(module, {"cta_mask": 2, "output": output})

    np.testing.assert_array_equal(result.outputs["output"], np.ones(1, dtype=np.uint32))
