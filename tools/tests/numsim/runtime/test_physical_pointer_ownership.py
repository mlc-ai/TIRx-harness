"""Regression coverage for reusable physical-pointer operands.

Physical-pointer bindings are reusable values: one binding may feed any
number of later operations (or several operand slots of a single call).
The existing owned `Address` ABI remains the engine boundary; reusable
`PhysicalPtr` values are cheap `Arc` clones, and derivation paths use
copy-on-write when they need private state.  The oracle is behavioral: each
kernel must transpile, build natively (the boundary where an accidental move
fails with rustc E0382), and — where it writes observable output — execute to
the expected values.
"""

from __future__ import annotations

import numpy as np

from tirx_harness import numsim
from tirx_harness.numsim.transpiler import suspend_scaffold
from tvm.backend.cuda.lang.clc import query_cancel_first_ctaid_x
from tvm.script import tirx as T

_PREFETCH_L2_SOURCE = r"""
__forceinline__ __device__ void tirx_prefetch_l2(const void* p) {
    asm volatile("prefetch.global.L2 [%0];" :: "l"(p));
}
"""

_ST_ASYNC_CLUSTER_TASK_INFO_SOURCE = r"""
__forceinline__ __device__ void tvm_builtin_st_async_cluster_task_info(
    void* dst, void* bar, uint32_t dst_cta_idx,
    uint32_t v0, uint32_t v1, uint32_t v2, uint32_t v3,
    uint32_t v4, uint32_t v5, uint32_t v6, uint32_t v7) {
    const uint32_t bar_addr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
    const uint32_t dst_addr = static_cast<uint32_t>(__cvta_generic_to_shared(dst));
    uint32_t mapped_bar, mapped_dst;
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;"
                 : "=r"(mapped_bar) : "r"(bar_addr), "r"(dst_cta_idx));
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;"
                 : "=r"(mapped_dst) : "r"(dst_addr), "r"(dst_cta_idx));
    asm volatile(
        "st.async.shared::cluster.mbarrier::complete_tx::bytes.u32.v4 [%0], {%1, %2, %3, %4}, [%5];" ::
        "r"(mapped_dst), "r"(v0), "r"(v1), "r"(v2), "r"(v3), "r"(mapped_bar));
    asm volatile(
        "st.async.shared::cluster.mbarrier::complete_tx::bytes.u32.v4 [%0], {%1, %2, %3, %4}, [%5];" ::
        "r"(mapped_dst + 16), "r"(v4), "r"(v5), "r"(v6), "r"(v7), "r"(mapped_bar));
}
"""


@T.prim_func
def reused_raw_ldmatrix_pointer(output: T.Buffer((32, 2), "uint32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((512,), "uint8", scope="shared")
    first = T.alloc_local((4,), "uint32")
    second = T.alloc_local((4,), "uint32")
    for byte in T.unroll(16):
        shared[lane * 16 + byte] = T.uint8(165)
    T.cuda.warp_sync()
    source_pointer = shared.ptr_to([lane * 16])

    T.ptx.ldmatrix.sync.aligned.m8n8.x4.shared.b16(
        first[0],
        first[1],
        first[2],
        first[3],
        source_pointer,
    )
    T.ptx.ldmatrix.sync.aligned.m8n8.x4.shared.b16(
        second[0],
        second[1],
        second[2],
        second[3],
        source_pointer,
    )
    output[lane, 0] = first[0]
    output[lane, 1] = second[0]


@T.prim_func
def reused_stmatrix_pointer():
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((8, 8), "uint16", scope="shared")
    source = T.alloc_buffer((1,), "uint32", scope="local")
    source[0] = T.cast(lane, "uint32")
    destination_pointer = shared.ptr_to([lane % 8, 0])

    T.ptx.stmatrix.sync.aligned.m8n8.x1.shared.b16(
        destination_pointer,
        source[0],
    )
    T.ptx.stmatrix.sync.aligned.m8n8.x1.shared.b16(
        destination_pointer,
        source[0],
    )


@T.prim_func
def reused_runtime_descriptor_pointer(descriptor: T.Buffer((32,), "uint32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    descriptor_pointer = descriptor.ptr_to([lane])

    T.cuda.runtime_instr_desc(descriptor_pointer, lane % 4)
    T.cuda.nano_sleep(T.uint32(1))
    T.cuda.runtime_instr_desc(descriptor_pointer, lane % 4)


@T.prim_func
def reused_legacy_mma_pointer_bindings(
    a_words: T.Buffer((32, 4), "uint32"),
    b_words: T.Buffer((32, 2), "uint32"),
    c_values: T.Buffer((32, 4), "float32"),
    output: T.Buffer((32, 4), "uint32"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    a = T.alloc_local((4,), "uint32")
    b = T.alloc_local((2,), "uint32")
    c = T.alloc_local((4,), "float32")
    d = T.alloc_local((4,), "float32")
    for register in T.unroll(4):
        a[register] = a_words[lane, register]
        c[register] = c_values[lane, register]
    for register in T.unroll(2):
        b[register] = b_words[lane, register]
    a_pointer = a.ptr_to([0])
    b_pointer = b.ptr_to([0])
    accumulator_pointer = d.ptr_to([0])

    # The legacy form uses one accumulator pointer for both C and D.  All
    # three pointer bindings are reused verbatim by the second call.
    d[0] = c[0]
    d[1] = c[0]
    d[2] = c[2]
    d[3] = c[3]
    T.ptx_legacy.mma(
        "m16n8k16",
        "row",
        "col",
        "float16",
        "float16",
        "float32",
        a_pointer,
        0,
        b_pointer,
        0,
        accumulator_pointer,
        0,
        False,
        dtype="float32",
    )
    d[0] = c[0]
    d[1] = c[0]
    d[2] = c[2]
    d[3] = c[3]
    T.ptx_legacy.mma(
        "m16n8k16",
        "row",
        "col",
        "float16",
        "float16",
        "float32",
        a_pointer,
        0,
        b_pointer,
        0,
        accumulator_pointer,
        0,
        False,
        dtype="float32",
    )
    for register in T.unroll(4):
        output[lane, register] = T.reinterpret("uint32", d[register])


@T.prim_func
def reused_mma_store_pointers(output: T.Buffer((16, 16), "float32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    fragment = T.alloc_local((8,), "float32")

    for local_id in T.serial(8):
        fragment[local_id] = T.cast(lane * 8 + local_id, "float32")

    destination_pointer = output.ptr_to([0, 0])
    source_pointer = fragment.ptr_to([0])
    T.evaluate(
        T.cuda.mma_store(16, 16, destination_pointer, source_pointer, 0, 16, dtype="float32")
    )
    T.evaluate(
        T.cuda.mma_store(16, 16, destination_pointer, source_pointer, 0, 16, dtype="float32")
    )


@T.prim_func
def reused_tcgen05_pointers(output: T.Buffer((2,), "uint32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    address = T.alloc_buffer((1,), "uint32", scope="shared")
    barrier = T.alloc_buffer((1,), "uint64", scope="shared")
    address_pointer = address.ptr_to([0])
    barrier_pointer = barrier.ptr_to([0])

    T.ptx.tcgen05.alloc.cta_group__1.sync.aligned.shared__cta.b32(address_pointer, 32)
    T.cuda.warp_sync()
    if lane == 0:
        T.ptx.mbarrier.init.shared.b64(barrier_pointer, 1)
    T.ptx.fence.proxy.async_.shared__cta()
    T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.cta_sync()
    if lane == 0:
        T.ptx.ld.shared.u32(output[0], address_pointer)
        T.ptx.tcgen05.commit.cta_group__1.mbarrier__arrive__one.shared__cluster.b64(barrier_pointer)
        T.ptx.mbarrier.test_wait.parity.shared.b64(output[1], barrier_pointer, 0)
    T.cuda.cta_sync()
    T.ptx.tcgen05.dealloc.cta_group__1.sync.aligned.b32(address[0], 32)
    T.ptx.tcgen05.relinquish_alloc_permit.cta_group__1.sync.aligned()


@T.prim_func
def reused_blocking_mbarrier_wait_pointer(output: T.Buffer((32,), "uint32")):
    T.device_entry()
    warp = T.warp_id([2])
    lane = T.lane_id([32])
    barriers = T.alloc_buffer((2,), "uint64", scope="shared", align=8)
    payload = T.alloc_buffer((32,), "uint32", scope="shared")
    produced_pointer = barriers.ptr_to([0])
    consumed_pointer = barriers.ptr_to([1])

    if warp == 0 and lane == 0:
        T.ptx.mbarrier.init.shared.b64(produced_pointer, 32)
        T.ptx.mbarrier.init.shared.b64(consumed_pointer, 32)
    T.ptx.fence.proxy.async_.shared__cta()
    T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.cta_sync()
    # Cross-wait: each warp arrives on its own barrier, then takes a blocking
    # wait on the other warp's barrier through a reused pointer binding
    # (init + arrive + wait all borrow the same binding).  Whichever warp
    # reaches its wait first genuinely suspends, so the awaited blocking-wait
    # path executes under every schedule.
    if warp == 0:
        payload[lane] = T.cast(lane * 7 + 3, "uint32")
        T.ptx.mbarrier.arrive.shared.b64(produced_pointer)
        T.cuda.mbarrier_wait(consumed_pointer, 0)
    else:
        T.ptx.mbarrier.arrive.shared.b64(consumed_pointer)
        T.cuda.mbarrier_wait(produced_pointer, 0)
        output[lane] = payload[lane]


@T.prim_func
def reused_scalar_wait_pointers(output: T.Buffer((32, 4), "uint32")):
    T.device_entry()
    _cluster = T.cluster_id([1])
    _cta = T.cta_id_in_cluster([1])
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    barrier = T.alloc_buffer((1,), "uint64", scope="shared")
    response = T.alloc_buffer((4,), "uint32", scope="shared", align=16)
    first_cancel = T.local_scalar("uint32")
    second_cancel = T.local_scalar("uint32")
    barrier_pointer = barrier.ptr_to([0])
    response_pointer = response.ptr_to([0])

    if lane == 0:
        T.ptx.mbarrier.init.shared.b64(barrier_pointer, 1)
    T.ptx.fence.proxy.async_.shared__cta()
    T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.cta_sync()
    T.ptx.mbarrier.test_wait.parity.shared.b64(output[lane, 0], barrier_pointer, 0)
    T.ptx.mbarrier.test_wait.parity.shared.b64(output[lane, 1], barrier_pointer, 0)
    if lane == 0:
        T.ptx.mbarrier.arrive.expect_tx.shared.b64(barrier_pointer, 16)
        T.ptx[
            "clusterlaunchcontrol.try_cancel.async.shared::cta"
            ".mbarrier::complete_tx::bytes.multicast::cluster::all.b128"
        ](response_pointer, barrier_pointer)
        T.cuda.mbarrier_wait_acquire_cluster(barrier_pointer, 0)
        query_cancel_first_ctaid_x(first_cancel, response_pointer)
        query_cancel_first_ctaid_x(second_cancel, response_pointer)
        output[0, 2] = first_cancel
        output[0, 3] = second_cancel


@T.prim_func
def reused_pointer_across_generic_consumers():
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    barrier = T.alloc_buffer((1,), "uint64", scope="shared", align=8)

    if lane == 0:
        T.ptx.mbarrier.init.shared.b64(T.address_of(barrier[0]), 2)
    T.ptx.fence.proxy.async_.shared__cta()
    T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.cta_sync()
    if lane == 0:
        pointer = barrier.ptr_to([0])
        selected: T.let[T.handle] = T.call_intrin(
            "handle",
            "prim.if_then_else",
            T.bool(True),
            pointer,
            pointer,
        )
        alias: T.let[T.handle] = selected
        T.cuda.printf("mbar=%p", pointer)
        T.cuda.smem_addr_from_uint64(pointer)
        T.ptx.mbarrier.arrive.shared.b64(alias)
        T.ptx.mbarrier.arrive.shared.b64(selected)
        T.cuda.mbarrier_wait(pointer, 0)


@T.prim_func
def reused_prefetch_l2_pointer(source: T.Buffer((32,), "uint32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    pointer = source.ptr_to([lane])

    T.evaluate(
        T.cuda.func_call(
            "tirx_prefetch_l2",
            pointer,
            source_code=_PREFETCH_L2_SOURCE,
            return_type="void",
        )
    )
    T.evaluate(
        T.cuda.func_call(
            "tirx_prefetch_l2",
            pointer,
            source_code=_PREFETCH_L2_SOURCE,
            return_type="void",
        )
    )


@T.prim_func
def reused_mapa_source_pointer():
    T.device_entry()
    _cluster = T.cluster_id([1])
    _cta = T.cta_id_in_cluster([1])
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((1,), "uint32", scope="shared")
    first = T.alloc_local((1,), "uint64")
    second = T.alloc_local((1,), "uint64")
    source_pointer = shared.ptr_to([0])

    if lane == 0:
        T.ptx.mapa.u64(first[0], source_pointer, T.uint32(0))
        T.ptx.mapa.u64(second[0], source_pointer, T.uint32(0))


@T.prim_func
def reused_st_async_cluster_task_info_pointers():
    T.device_entry()
    _cluster = T.cluster_id([1])
    _cta = T.cta_id_in_cluster([1])
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    destination = T.alloc_buffer((8,), "uint32", scope="shared", align=16)
    barrier = T.alloc_buffer((1,), "uint64", scope="shared", align=8)
    destination_pointer = destination.ptr_to([0])
    barrier_pointer = barrier.ptr_to([0])

    if lane == 0:
        for _repeat in T.unroll(2):
            T.evaluate(
                T.cuda.func_call(
                    "tvm_builtin_st_async_cluster_task_info",
                    destination_pointer,
                    barrier_pointer,
                    T.uint32(0),
                    T.uint32(1),
                    T.uint32(2),
                    T.uint32(3),
                    T.uint32(4),
                    T.uint32(5),
                    T.uint32(6),
                    T.uint32(7),
                    T.uint32(8),
                    source_code=_ST_ASYNC_CLUSTER_TASK_INFO_SOURCE,
                    return_type="void",
                )
            )


@T.prim_func
def one_shot_temporary_pointer_operand(output: T.Buffer((32,), "uint32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((32,), "uint32", scope="shared")
    shared[lane] = T.cast(lane * 3 + 1, "uint32")
    T.ptx.ld.shared.u32(output[lane], shared.ptr_to([lane]))


_REUSED_POINTER_CASES = (
    reused_raw_ldmatrix_pointer,
    reused_stmatrix_pointer,
    reused_runtime_descriptor_pointer,
    reused_legacy_mma_pointer_bindings,
    reused_mma_store_pointers,
    reused_tcgen05_pointers,
    reused_blocking_mbarrier_wait_pointer,
    reused_scalar_wait_pointers,
    reused_pointer_across_generic_consumers,
    reused_prefetch_l2_pointer,
    reused_mapa_source_pointer,
    reused_st_async_cluster_task_info_pointers,
)


def test_reused_physical_pointers_build_one_native_artifact(tmp_path):
    # The native build is the oracle: an accidental move of a reused pointer
    # binding fails rustc borrow checking (E0382) inside this build.
    numsim.transpile(_REUSED_POINTER_CASES, cache_dir=tmp_path)


def test_reused_pointer_builds_and_runs_across_a_forced_root_async_split(tmp_path, monkeypatch):
    monkeypatch.setattr(suspend_scaffold, "_ROOT_ASYNC_SPLIT_MIN_LINES", 0)
    module = numsim.transpile(reused_runtime_descriptor_pointer, cache_dir=tmp_path)

    numsim.Engine().run(module, {"descriptor": np.zeros(32, dtype=np.uint32)})


def test_reused_raw_ldmatrix_pointer_executes_both_loads(tmp_path):
    module = numsim.transpile(reused_raw_ldmatrix_pointer, cache_dir=tmp_path)

    result = numsim.Engine().run(module, {"output": np.zeros((32, 2), dtype=np.uint32)})

    # Every shared byte is 0xA5, so any fragment word either load distributes
    # is 0xA5A5A5A5; both loads through the shared source binding must land.
    np.testing.assert_array_equal(
        result.outputs["output"], np.full((32, 2), 0xA5A5A5A5, dtype=np.uint32)
    )


def test_reused_legacy_mma_reuses_a_b_and_accumulator_bindings_across_calls(tmp_path):
    packed_fp16_ones = np.uint32(0x3C003C00)
    a_words = np.full((32, 4), packed_fp16_ones, dtype=np.uint32)
    b_words = np.full((32, 2), packed_fp16_ones, dtype=np.uint32)
    c_values = np.arange(32 * 4, dtype=np.float32).reshape(32, 4) / 4

    module = numsim.transpile(reused_legacy_mma_pointer_bindings, cache_dir=tmp_path)
    result = numsim.Engine().run(
        module,
        {
            "a_words": a_words,
            "b_words": b_words,
            "c_values": c_values,
            "output": np.zeros((32, 4), dtype=np.uint32),
        },
    )

    # With every A/B half equal to 1.0 each product fragment is exactly 16.0,
    # so d[slot] = 16.0 + c[slot] independently of the lane->element mapping.
    # The second accumulator register is seeded from c[0], so d[1] must
    # observe c[0].
    accumulators = c_values.copy()
    accumulators[:, 1] = c_values[:, 0]
    expected = (np.float32(16.0) + accumulators).view(np.uint32)
    np.testing.assert_array_equal(result.outputs["output"], expected)


def test_reused_mma_store_pointers_write_lane_register_layout(tmp_path):
    module = numsim.transpile(reused_mma_store_pointers, cache_dir=tmp_path)

    result = numsim.Engine().run(module, {"output": np.zeros((16, 16), dtype=np.float32)})

    expected = np.empty((16, 16), dtype=np.float32)
    for row in range(16):
        for column in range(16):
            lane = 4 * (row % 8) + (column % 8) // 2
            local_id = 4 * (column // 8) + 2 * (row // 8) + column % 2
            expected[row, column] = lane * 8 + local_id
    np.testing.assert_array_equal(result.outputs["output"], expected)


def test_reused_tcgen05_pointers_execute_alloc_store_and_commit(tmp_path):
    module = numsim.transpile(reused_tcgen05_pointers, cache_dir=tmp_path)

    result = numsim.Engine().run(module, {"output": np.full(2, 0xDEAD, dtype=np.uint32)})

    # The first allocation in an empty TMEM starts at base column 0, and the
    # architected alloc store publishes exactly that address through the
    # shared destination binding (the awaited alloc path).  tcgen05.commit
    # completes its mbarrier asynchronously: hardware may complete it at any
    # point, and NumSim's canonical schedule completes the numeric
    # transaction immediately (there is no separate completion actor to
    # await), so the nonblocking parity-0 query may observe either phase.
    # Both outcomes are distinct from the 0xDEAD sentinel, so both stores
    # ran through the reused pointers.
    output = result.outputs["output"]
    np.testing.assert_array_equal(output[0], np.uint32(0))
    assert output[1] in (np.uint32(0), np.uint32(1)), output[1]


def test_reused_blocking_mbarrier_wait_pointer_executes_across_the_wait(tmp_path):
    module = numsim.transpile(reused_blocking_mbarrier_wait_pointer, cache_dir=tmp_path)

    result = numsim.Engine().run(module, {"output": np.zeros(32, dtype=np.uint32)})

    # Warp 1's parity-0 wait completes only after warp 0's 32 arrivals, so
    # the payload warp 0 published before arriving is fully visible; the
    # cross-wait guarantees one side genuinely suspended, and termination
    # proves both blocking waits resumed.
    np.testing.assert_array_equal(result.outputs["output"], np.arange(32, dtype=np.uint32) * 7 + 3)


def test_one_shot_temporary_pointer_operand_builds_and_runs(tmp_path):
    module = numsim.transpile(one_shot_temporary_pointer_operand, cache_dir=tmp_path)

    result = numsim.Engine().run(module, {"output": np.zeros(32, dtype=np.uint32)})

    np.testing.assert_array_equal(result.outputs["output"], np.arange(32, dtype=np.uint32) * 3 + 1)
