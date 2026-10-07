from __future__ import annotations

import numpy as np
import pytest
from tvm.ir.type import PointerType, PrimType
from tvm.script import tirx as T

from tirx_harness import numsim
from tirx_harness.numsim.cases import _descriptor_storage
from tirx_harness.numsim.errors import UnsupportedTIRxError


_FLASHKDA_ACQUIRE_SOURCE = r"""__device__ __forceinline__ void flashkda_tensormap_acquire(const void *tmap_ptr) {
    asm volatile(
        "fence.proxy.tensormap::generic.acquire.gpu [%0], 128;\n"
        :: "l"(tmap_ptr) : "memory");
}
"""

_GDN_REPLACE_ADDRESS_SOURCE = r"""__device__ __forceinline__ void gdn_tensormap_replace_global_address(void *desc, const void *addr) {
    asm volatile("tensormap.replace.tile.global_address.global.b1024.b64 [%0], %1;"
                 :: "l"(desc), "l"(addr) : "memory");
}
"""


def _gdn_replace_dim_source(index: int) -> str:
    return f"""__device__ __forceinline__ void gdn_tensormap_replace_global_dim_{index}(void *desc, unsigned int value) {{
    asm volatile("tensormap.replace.tile.global_dim.global.b1024.b32 [%0], {index}, %1;"
                 :: "l"(desc), "r"(value) : "memory");
}}
"""


def _gdn_replace_stride_source(index: int) -> str:
    return f"""__device__ __forceinline__ void gdn_tensormap_replace_global_stride_{index}(void *desc, unsigned long long value) {{
    asm volatile("tensormap.replace.tile.global_stride.global.b1024.b64 [%0], {index}, %1;"
                 :: "l"(desc), "l"(value) : "memory");
}}
"""


_GDN_RELEASE_SOURCE = r"""__device__ __forceinline__ void gdn_tensormap_release() {
    asm volatile("fence.proxy.tensormap::generic.release.gpu;" ::: "memory");
}
"""

_GDN_ACQUIRE_SOURCE = r"""__device__ __forceinline__ void gdn_tensormap_acquire(const void *desc) {
    asm volatile("fence.proxy.tensormap::generic.acquire.gpu [%0], 128;"
                 :: "l"(desc) : "memory");
}
"""


@T.inline
def _descriptor_tma_load_body(descriptor, shared, barriers, output):
    T.evaluate(
        T.ptx[
            "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.cta_group::1"
        ](T.address_of(shared[0, 0]), descriptor, 0, 0, T.address_of(barriers[0]))
    )
    T.ptx.mbarrier.arrive.expect_tx.shared.b64(T.address_of(barriers[0]), 48)
    T.cuda.mbarrier_wait(T.address_of(barriers[0]), 0)
    for row in T.serial(3):
        for column in T.serial(4):
            output[row, column] = shared[row, column]


def _replace_global_dim(descriptor, index: int, value):
    return T.cuda.func_call(
        f"gdn_tensormap_replace_global_dim_{index}",
        descriptor,
        value,
        source_code=_gdn_replace_dim_source(index),
        return_type="void",
    )


def _replace_global_stride(descriptor, index: int, value):
    return T.cuda.func_call(
        f"gdn_tensormap_replace_global_stride_{index}",
        descriptor,
        value,
        source_code=_gdn_replace_stride_source(index),
        return_type="void",
    )


@T.inline
def _copy_descriptor_payload(source_map, descriptor):
    payload = T.decl_buffer(
        (8,),
        "uint64",
        data=T.reinterpret("handle", T.address_of(source_map)),
        scope="param",
        align=16,
    )
    T.ptx.st.global_.v4.b64(descriptor, payload[0], payload[1], payload[2], payload[3])
    T.ptx.st.global_.v4.b64(
        T.reinterpret("handle", T.reinterpret("uint64", descriptor) + T.uint64(32)),
        payload[4],
        payload[5],
        payload[6],
        payload[7],
    )


@T.inline
def _copy_raw_descriptor_payload_with_scalar_stores(source_descriptor_storage, descriptor):
    payload = T.alloc_buffer((4,), "uint64", scope="local")
    source = T.reinterpret("uint64", source_descriptor_storage.ptr_to([0]))
    target = T.reinterpret("uint64", descriptor)
    for group in range(2):
        offset = T.uint64(group * 32)
        T.ptx.ld.global_.v4.b64(
            payload[0],
            payload[1],
            payload[2],
            payload[3],
            T.reinterpret("handle", source + offset),
        )
        for word in range(4):
            T.ptx.st.global_.b64(
                T.reinterpret("handle", target + T.uint64((group * 4 + word) * 8)),
                payload[word],
            )


@T.prim_func
def typed_param_descriptor_scalar_copy(
    source_map: T.TensorMap(), output: T.Buffer((16,), "uint64")
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    payload = T.decl_buffer(
        (16,),
        "uint64",
        data=T.reinterpret(PointerType(PrimType("uint64"), "param"), T.address_of(source_map)),
        scope="param",
    )
    if lane == 0:
        for word in T.serial(16):
            T.ptx.st.global_.b64(output.ptr_to([word]), payload[word])


def test_typed_param_descriptor_scalar_copy_preserves_payload(tmp_path):
    source = np.arange(12, dtype=np.float32).reshape(3, 4)
    descriptor = _tensor_map(source)
    module = numsim.transpile(typed_param_descriptor_scalar_copy, cache_dir=tmp_path)
    result = numsim.Engine().run(
        module,
        {"source_map": descriptor, "output": np.zeros(16, dtype=np.uint64)},
        outputs=("output",),
    )
    np.testing.assert_array_equal(result.outputs["output"], descriptor.view(np.uint64))
    assert result.verdict == "clean"


@T.prim_func
def host_descriptor_tma_load(
    source: T.Buffer((3, 4), "float32"),
    output: T.Buffer((3, 4), "float32"),
    descriptor_storage: T.Buffer((16,), "int64"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((3, 4), "float32", scope="shared")
    barriers = T.alloc_buffer((1,), "uint64", scope="shared")
    descriptor = descriptor_storage.ptr_to([0])
    if lane == 0:
        T.cuda.func_call(
            "flashkda_tensormap_acquire",
            descriptor,
            source_code=_FLASHKDA_ACQUIRE_SOURCE,
            return_type="void",
        )
        T.ptx.mbarrier.init.shared.b64(T.address_of(barriers[0]), 1)
    T.ptx.fence.proxy.async_.shared__cta()
    T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.cta_sync()
    if lane == 0:
        _descriptor_tma_load_body(descriptor, shared, barriers, output)


@T.prim_func
def host_descriptor_tma_load_without_acquire(
    source: T.Buffer((3, 4), "float32"),
    output: T.Buffer((3, 4), "float32"),
    descriptor_storage: T.Buffer((16,), "int64"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((3, 4), "float32", scope="shared")
    barriers = T.alloc_buffer((1,), "uint64", scope="shared")
    descriptor = descriptor_storage.ptr_to([0])
    if lane == 0:
        T.ptx.mbarrier.init.shared.b64(T.address_of(barriers[0]), 1)
    T.ptx.fence.proxy.async_.shared__cta()
    T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.cta_sync()
    if lane == 0:
        _descriptor_tma_load_body(descriptor, shared, barriers, output)


@T.prim_func
def host_descriptor_tma_load_to_raw_shared_offset(
    source: T.Buffer((3, 4), "float32"),
    output: T.Buffer((3, 4), "float32"),
    descriptor_storage: T.Buffer((16,), "int64"),
    destination_row: T.int32,
    barrier_index: T.int32,
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((11, 4), "float32", scope="shared")
    barriers = T.alloc_buffer((2,), "uint64", scope="shared")
    descriptor = descriptor_storage.ptr_to([0])
    shared_base: T.uint32 = T.cuda.cvta_generic_to_shared(shared.ptr_to([0, 0]))
    destination = shared_base + T.cast(destination_row * 16, "uint32")
    barrier_base: T.uint32 = T.cuda.cvta_generic_to_shared(barriers.ptr_to([0]))
    transaction_barrier = barrier_base + T.cast(barrier_index * 8, "uint32")
    if lane == 0:
        T.cuda.func_call(
            "flashkda_tensormap_acquire",
            descriptor,
            source_code=_FLASHKDA_ACQUIRE_SOURCE,
            return_type="void",
        )
        T.ptx.mbarrier.init.shared.b64(T.address_of(barriers[barrier_index]), 1)
    T.ptx.fence.proxy.async_.shared__cta()
    T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.cta_sync()
    if lane == 0:
        T.evaluate(
            T.ptx[
                "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.cta_group::1"
            ](destination, descriptor, 0, 0, transaction_barrier)
        )
        T.ptx.mbarrier.arrive.expect_tx.shared.b64(
            barrier_base + T.cast(barrier_index * 8, "uint32"), 48
        )
        T.cuda.mbarrier_wait(T.address_of(barriers[barrier_index]), 0)
        for row in T.serial(3):
            for column in T.serial(4):
                output[row, column] = shared[row + destination_row, column]


@T.prim_func
def copied_and_replaced_descriptor_tma_load(
    source_map: T.TensorMap(),
    source: T.Buffer((3, 4), "float32"),
    replacement: T.Buffer((3, 4), "float32"),
    output: T.Buffer((3, 4), "float32"),
    descriptor_storage: T.Buffer((128,), "uint8"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((3, 4), "float32", scope="shared")
    barriers = T.alloc_buffer((1,), "uint64", scope="shared")
    descriptor = descriptor_storage.ptr_to([0])
    if lane == 0:
        payload = T.decl_buffer(
            (8,),
            "uint64",
            data=T.reinterpret("handle", T.address_of(source_map)),
            scope="param",
            align=16,
        )
        T.ptx.st.global_.v4.b64(descriptor, payload[0], payload[1], payload[2], payload[3])
        T.ptx.st.global_.v4.b64(
            T.reinterpret("handle", T.reinterpret("uint64", descriptor) + T.uint64(32)),
            payload[4],
            payload[5],
            payload[6],
            payload[7],
        )
        T.cuda.func_call(
            "gdn_tensormap_replace_global_address",
            descriptor,
            replacement.data,
            source_code=_GDN_REPLACE_ADDRESS_SOURCE,
            return_type="void",
        )
        _replace_global_dim(descriptor, 0, T.uint32(4))
        _replace_global_dim(descriptor, 1, T.uint32(3))
        _replace_global_dim(descriptor, 2, T.uint32(1))
        _replace_global_dim(descriptor, 3, T.uint32(1))
        _replace_global_dim(descriptor, 4, T.uint32(1))
        _replace_global_stride(descriptor, 0, T.uint64(16))
        _replace_global_stride(descriptor, 1, T.uint64(0))
        _replace_global_stride(descriptor, 2, T.uint64(0))
        _replace_global_stride(descriptor, 3, T.uint64(0))
        T.cuda.func_call(
            "gdn_tensormap_release",
            source_code=_GDN_RELEASE_SOURCE,
            return_type="void",
        )
        T.cuda.func_call(
            "gdn_tensormap_acquire",
            descriptor,
            source_code=_GDN_ACQUIRE_SOURCE,
            return_type="void",
        )
        T.ptx.mbarrier.init.shared.b64(T.address_of(barriers[0]), 1)
    T.ptx.fence.proxy.async_.shared__cta()
    T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.cta_sync()
    if lane == 0:
        _descriptor_tma_load_body(descriptor, shared, barriers, output)


@T.prim_func
def raw_copied_and_replaced_descriptor_tma_load(
    source_descriptor_storage: T.Buffer((16,), "int64"),
    source: T.Buffer((3, 4), "float32"),
    replacement: T.Buffer((3, 4), "float32"),
    output: T.Buffer((3, 4), "float32"),
    descriptor_workspace: T.Buffer((16,), "int64"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    payload = T.alloc_buffer((4,), "uint64", scope="local")
    shared = T.alloc_buffer((3, 4), "float32", scope="shared")
    barriers = T.alloc_buffer((1,), "uint64", scope="shared")
    source_descriptor = T.reinterpret("uint64", source_descriptor_storage.ptr_to([0]))
    descriptor = descriptor_workspace.ptr_to([0])
    destination_descriptor = T.reinterpret("uint64", descriptor)
    if lane == 0:
        for chunk in T.serial(2):
            byte_offset = T.cast(chunk * 32, "uint64")
            T.ptx.ld.global_.v4.b64(
                payload[0],
                payload[1],
                payload[2],
                payload[3],
                T.reinterpret("handle", source_descriptor + byte_offset),
            )
            T.ptx.st.global_.v4.b64(
                T.reinterpret("handle", destination_descriptor + byte_offset),
                payload[0],
                payload[1],
                payload[2],
                payload[3],
            )
        T.cuda.func_call(
            "gdn_tensormap_replace_global_address",
            descriptor,
            replacement.data,
            source_code=_GDN_REPLACE_ADDRESS_SOURCE,
            return_type="void",
        )
        _replace_global_dim(descriptor, 0, T.uint32(4))
        _replace_global_dim(descriptor, 1, T.uint32(3))
        _replace_global_dim(descriptor, 2, T.uint32(1))
        _replace_global_dim(descriptor, 3, T.uint32(1))
        _replace_global_dim(descriptor, 4, T.uint32(1))
        _replace_global_stride(descriptor, 0, T.uint64(16))
        _replace_global_stride(descriptor, 1, T.uint64(0))
        _replace_global_stride(descriptor, 2, T.uint64(0))
        _replace_global_stride(descriptor, 3, T.uint64(0))
        T.cuda.func_call(
            "gdn_tensormap_release",
            source_code=_GDN_RELEASE_SOURCE,
            return_type="void",
        )
        T.cuda.func_call(
            "gdn_tensormap_acquire",
            descriptor,
            source_code=_GDN_ACQUIRE_SOURCE,
            return_type="void",
        )
        T.ptx.mbarrier.init.shared.b64(T.address_of(barriers[0]), 1)
    T.ptx.fence.proxy.async_.shared__cta()
    T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.cta_sync()
    if lane == 0:
        _descriptor_tma_load_body(descriptor, shared, barriers, output)


@T.prim_func
def raw_descriptor_scalar_store_copy_tma_load(
    source_descriptor_storage: T.Buffer((16,), "int64"),
    output: T.Buffer((3, 4), "float32"),
    descriptor_workspace: T.Buffer((16,), "int64"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((3, 4), "float32", scope="shared")
    barriers = T.alloc_buffer((1,), "uint64", scope="shared")
    descriptor = descriptor_workspace.ptr_to([0])
    if lane == 0:
        _copy_raw_descriptor_payload_with_scalar_stores(source_descriptor_storage, descriptor)
        T.cuda.func_call(
            "gdn_tensormap_release",
            source_code=_GDN_RELEASE_SOURCE,
            return_type="void",
        )
        T.cuda.func_call(
            "gdn_tensormap_acquire",
            descriptor,
            source_code=_GDN_ACQUIRE_SOURCE,
            return_type="void",
        )
        T.ptx.mbarrier.init.shared.b64(T.address_of(barriers[0]), 1)
    T.ptx.fence.proxy.async_.shared__cta()
    T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.cta_sync()
    if lane == 0:
        _descriptor_tma_load_body(descriptor, shared, barriers, output)


@T.prim_func
def incomplete_raw_descriptor_copy_tma_load(
    source_descriptor_storage: T.Buffer((16,), "int64"),
    output: T.Buffer((3, 4), "float32"),
    descriptor_workspace: T.Buffer((16,), "int64"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    payload = T.alloc_buffer((4,), "uint64", scope="local")
    shared = T.alloc_buffer((3, 4), "float32", scope="shared")
    barriers = T.alloc_buffer((1,), "uint64", scope="shared")
    source_descriptor = T.reinterpret("uint64", source_descriptor_storage.ptr_to([0]))
    descriptor = descriptor_workspace.ptr_to([0])
    destination_descriptor = T.reinterpret("uint64", descriptor)
    if lane == 0:
        for chunk in T.serial(1):
            byte_offset = T.cast(chunk * 32, "uint64")
            T.ptx.ld.global_.v4.b64(
                payload[0],
                payload[1],
                payload[2],
                payload[3],
                T.reinterpret("handle", source_descriptor + byte_offset),
            )
            T.ptx.st.global_.v4.b64(
                T.reinterpret("handle", destination_descriptor + byte_offset),
                payload[0],
                payload[1],
                payload[2],
                payload[3],
            )
        T.cuda.func_call(
            "gdn_tensormap_release",
            source_code=_GDN_RELEASE_SOURCE,
            return_type="void",
        )
        T.cuda.func_call(
            "gdn_tensormap_acquire",
            descriptor,
            source_code=_GDN_ACQUIRE_SOURCE,
            return_type="void",
        )
        T.ptx.mbarrier.init.shared.b64(T.address_of(barriers[0]), 1)
    T.ptx.fence.proxy.async_.shared__cta()
    T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.cta_sync()
    if lane == 0:
        _descriptor_tma_load_body(descriptor, shared, barriers, output)


@T.prim_func
def raw_descriptor_96b_copy_tma_load(
    source_descriptor_storage: T.Buffer((16,), "int64"),
    output: T.Buffer((3, 4), "float32"),
    descriptor_workspace: T.Buffer((16,), "int64"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    payload = T.alloc_buffer((4,), "uint64", scope="local")
    shared = T.alloc_buffer((3, 4), "float32", scope="shared")
    barriers = T.alloc_buffer((1,), "uint64", scope="shared")
    source_descriptor = T.reinterpret("uint64", source_descriptor_storage.ptr_to([0]))
    descriptor = descriptor_workspace.ptr_to([0])
    destination_descriptor = T.reinterpret("uint64", descriptor)
    if lane == 0:
        for chunk in T.serial(3):
            byte_offset = T.cast(chunk * 32, "uint64")
            T.ptx.ld.global_.v4.b64(
                payload[0],
                payload[1],
                payload[2],
                payload[3],
                T.reinterpret("handle", source_descriptor + byte_offset),
            )
            T.ptx.st.global_.v4.b64(
                T.reinterpret("handle", destination_descriptor + byte_offset),
                payload[0],
                payload[1],
                payload[2],
                payload[3],
            )
        T.cuda.func_call(
            "gdn_tensormap_release",
            source_code=_GDN_RELEASE_SOURCE,
            return_type="void",
        )
        T.cuda.func_call(
            "gdn_tensormap_acquire",
            descriptor,
            source_code=_GDN_ACQUIRE_SOURCE,
            return_type="void",
        )
        T.ptx.mbarrier.init.shared.b64(T.address_of(barriers[0]), 1)
    T.ptx.fence.proxy.async_.shared__cta()
    T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.cta_sync()
    if lane == 0:
        _descriptor_tma_load_body(descriptor, shared, barriers, output)


@T.prim_func
def ordinary_u64x4_copy(source: T.Buffer((16,), "int64"), output: T.Buffer((16,), "int64")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    payload = T.alloc_buffer((4,), "uint64", scope="local")
    source_address = T.reinterpret("uint64", source.ptr_to([0]))
    output_address = T.reinterpret("uint64", output.ptr_to([0]))
    if lane == 0:
        for chunk in T.serial(4):
            byte_offset = T.cast(chunk * 32, "uint64")
            T.ptx.ld.global_.v4.b64(
                payload[0],
                payload[1],
                payload[2],
                payload[3],
                T.reinterpret("handle", source_address + byte_offset),
            )
            T.ptx.st.global_.v4.b64(
                T.reinterpret("handle", output_address + byte_offset),
                payload[0],
                payload[1],
                payload[2],
                payload[3],
            )


@T.prim_func
def copied_and_replaced_descriptor_tma_store(
    output_map: T.TensorMap(),
    source: T.Buffer((3, 4), "float32"),
    output: T.Buffer((3, 4), "float32"),
    descriptor_storage: T.Buffer((128,), "uint8"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((3, 4), "float32", scope="shared")
    descriptor = descriptor_storage.ptr_to([0])
    if lane < 12:
        shared[lane // 4, lane % 4] = source[lane // 4, lane % 4]
    T.cuda.warp_sync()
    if lane == 0:
        _copy_descriptor_payload(output_map, descriptor)
        T.cuda.func_call(
            "gdn_tensormap_replace_global_address",
            descriptor,
            output.data,
            source_code=_GDN_REPLACE_ADDRESS_SOURCE,
            return_type="void",
        )
        T.cuda.func_call(
            "gdn_tensormap_release",
            source_code=_GDN_RELEASE_SOURCE,
            return_type="void",
        )
        T.cuda.func_call(
            "gdn_tensormap_acquire",
            descriptor,
            source_code=_GDN_ACQUIRE_SOURCE,
            return_type="void",
        )
    T.cuda.warp_sync()
    T.ptx.fence.proxy.async_.shared__cta()
    if lane == 0:
        T.evaluate(
            T.ptx["cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group"](
                descriptor, 0, 0, T.address_of(shared[0, 0])
            )
        )
        T.ptx.cp.async_.bulk.commit_group()
        T.ptx.cp.async_.bulk.wait_group(0)


@T.prim_func
def host_descriptor_tma_store(
    source: T.Buffer((3, 4), "float32"),
    output: T.Buffer((3, 4), "float32"),
    descriptor_storage: T.Buffer((16,), "int64"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((3, 4), "float32", scope="shared")
    descriptor = descriptor_storage.ptr_to([0])
    if lane < 12:
        shared[lane // 4, lane % 4] = source[lane // 4, lane % 4]
    T.cuda.warp_sync()
    if lane == 0:
        T.cuda.func_call(
            "gdn_tensormap_acquire",
            descriptor,
            source_code=_GDN_ACQUIRE_SOURCE,
            return_type="void",
        )
    T.cuda.warp_sync()
    T.ptx.fence.proxy.async_.shared__cta()
    if lane == 0:
        T.evaluate(
            T.ptx["cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group"](
                descriptor, 0, 0, T.address_of(shared[0, 0])
            )
        )
        T.ptx.cp.async_.bulk.commit_group()
        T.ptx.cp.async_.bulk.wait_group(0)


@T.prim_func
def stale_generation_descriptor_tma_load(
    source_map: T.TensorMap(),
    output: T.Buffer((3, 4), "float32"),
    descriptor_storage: T.Buffer((128,), "uint8"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((3, 4), "float32", scope="shared")
    barriers = T.alloc_buffer((1,), "uint64", scope="shared")
    descriptor = descriptor_storage.ptr_to([0])
    if lane == 0:
        _copy_descriptor_payload(source_map, descriptor)
        T.cuda.func_call(
            "gdn_tensormap_release",
            source_code=_GDN_RELEASE_SOURCE,
            return_type="void",
        )
        T.cuda.func_call(
            "gdn_tensormap_acquire",
            descriptor,
            source_code=_GDN_ACQUIRE_SOURCE,
            return_type="void",
        )
        _replace_global_dim(descriptor, 0, T.uint32(4))
        T.cuda.func_call(
            "gdn_tensormap_release",
            source_code=_GDN_RELEASE_SOURCE,
            return_type="void",
        )
        T.ptx.mbarrier.init.shared.b64(T.address_of(barriers[0]), 1)
    T.ptx.fence.proxy.async_.shared__cta()
    T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.cta_sync()
    if lane == 0:
        _descriptor_tma_load_body(descriptor, shared, barriers, output)


@T.prim_func
def incomplete_typed_descriptor_copy_tma_load(
    source_map: T.TensorMap(),
    output: T.Buffer((3, 4), "float32"),
    descriptor_storage: T.Buffer((128,), "uint8"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((3, 4), "float32", scope="shared")
    barriers = T.alloc_buffer((1,), "uint64", scope="shared")
    descriptor = descriptor_storage.ptr_to([0])
    if lane == 0:
        payload = T.decl_buffer(
            (8,),
            "uint64",
            data=T.reinterpret("handle", T.address_of(source_map)),
            scope="param",
            align=16,
        )
        T.ptx.st.global_.v4.b64(descriptor, payload[0], payload[1], payload[2], payload[3])
        T.cuda.func_call(
            "gdn_tensormap_release",
            source_code=_GDN_RELEASE_SOURCE,
            return_type="void",
        )
        T.cuda.func_call(
            "gdn_tensormap_acquire",
            descriptor,
            source_code=_GDN_ACQUIRE_SOURCE,
            return_type="void",
        )
        T.ptx.mbarrier.init.shared.b64(T.address_of(barriers[0]), 1)
    T.ptx.fence.proxy.async_.shared__cta()
    T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.cta_sync()
    if lane == 0:
        _descriptor_tma_load_body(descriptor, shared, barriers, output)


@T.prim_func
def maximum_dimension_descriptor_update(
    source_map: T.TensorMap(), descriptor_storage: T.Buffer((128,), "uint8")
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    descriptor = descriptor_storage.ptr_to([0])
    if lane == 0:
        _copy_descriptor_payload(source_map, descriptor)
        _replace_global_dim(descriptor, 0, T.uint32(0))
        T.cuda.func_call(
            "gdn_tensormap_release",
            source_code=_GDN_RELEASE_SOURCE,
            return_type="void",
        )


@T.prim_func
def invalid_dimension_index_descriptor_update(
    source_map: T.TensorMap(), descriptor_storage: T.Buffer((128,), "uint8")
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    descriptor = descriptor_storage.ptr_to([0])
    if lane == 0:
        _copy_descriptor_payload(source_map, descriptor)
        _replace_global_dim(descriptor, 5, T.uint32(1))


@T.prim_func
def conflicting_host_descriptor_slots(
    output: T.Buffer((3, 4), "float32"),
    descriptor_storage_a: T.Buffer((128,), "uint8"),
    descriptor_storage_b: T.Buffer((128,), "uint8"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((3, 4), "float32", scope="shared")
    barriers = T.alloc_buffer((1,), "uint64", scope="shared")
    descriptor = descriptor_storage_a.ptr_to([0])
    if lane == 0:
        T.cuda.func_call(
            "flashkda_tensormap_acquire",
            descriptor,
            source_code=_FLASHKDA_ACQUIRE_SOURCE,
            return_type="void",
        )
        T.ptx.mbarrier.init.shared.b64(T.address_of(barriers[0]), 1)
    T.ptx.fence.proxy.async_.shared__cta()
    T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.cta_sync()
    if lane == 0:
        _descriptor_tma_load_body(descriptor, shared, barriers, output)


def _tensor_map(
    base: np.ndarray,
) -> np.ndarray:
    return numsim.TensorMap(
        base=base,
        global_shape=(4, 3),
        global_strides=(16,),
        box_shape=(4, 3),
        element_strides=(1, 1),
    ).numpy()


def _host_descriptor_bindings(source: np.ndarray, output: np.ndarray):
    return {
        "source": source,
        "output": output,
        "descriptor_storage": _descriptor_storage(
            storage=np.zeros(16, dtype=np.int64),
            slots={0: _tensor_map(source)},
        ),
    }


def test_host_bound_descriptor_acquire_enables_raw_tma(tmp_path):
    source = np.arange(12, dtype=np.float32).reshape(3, 4)
    output = np.zeros_like(source)
    module = numsim.transpile(host_descriptor_tma_load, cache_dir=tmp_path)

    result = numsim.Engine().run(
        module,
        _host_descriptor_bindings(source, output),
        outputs=("output",),
    )

    np.testing.assert_array_equal(result.outputs["output"], source)


def test_host_bound_descriptor_without_acquire_fails_closed(tmp_path):
    source = np.arange(12, dtype=np.float32).reshape(3, 4)
    output = np.zeros_like(source)
    module = numsim.transpile(host_descriptor_tma_load_without_acquire, cache_dir=tmp_path)

    with pytest.raises(numsim.NumSimExecutionError, match="not acquired"):
        numsim.Engine().run(
            module,
            _host_descriptor_bindings(source, output),
            outputs=("output",),
        )


def test_zero_descriptor_bytes_fail_image_validation(tmp_path):
    source = np.arange(12, dtype=np.float32).reshape(3, 4)
    output = np.zeros_like(source)
    module = numsim.transpile(host_descriptor_tma_load, cache_dir=tmp_path)

    with pytest.raises(numsim.NumSimExecutionError, match="image has invalid magic"):
        numsim.Engine().run(
            module,
            {
                "source": source,
                "output": output,
                "descriptor_storage": np.zeros(16, dtype=np.int64),
            },
            outputs=("output",),
        )


def test_raw_shared_address_binding_and_offset_preserve_tma_destination(tmp_path):
    source = np.arange(12, dtype=np.float32).reshape(3, 4)
    output = np.zeros_like(source)
    module = numsim.transpile(
        host_descriptor_tma_load_to_raw_shared_offset,
        cache_dir=tmp_path,
    )

    result = numsim.Engine().run(
        module,
        {
            **_host_descriptor_bindings(source, output),
            "destination_row": 8,
            "barrier_index": 1,
        },
        outputs=("output",),
    )

    np.testing.assert_array_equal(result.outputs["output"], source)


def test_raw_shared_address_outside_declared_views_fails_closed(tmp_path):
    source = np.arange(12, dtype=np.float32).reshape(3, 4)
    output = np.zeros_like(source)
    module = numsim.transpile(
        host_descriptor_tma_load_to_raw_shared_offset,
        cache_dir=tmp_path,
    )

    with pytest.raises(
        numsim.NumSimExecutionError,
        match="does not name a declared shared-memory view",
    ):
        numsim.Engine().run(
            module,
            {
                **_host_descriptor_bindings(source, output),
                "destination_row": 100,
                "barrier_index": 1,
            },
            outputs=("output",),
        )


def test_typed_payload_copy_replace_release_and_acquire_drive_raw_tma(tmp_path):
    source = np.arange(12, dtype=np.float32).reshape(3, 4)
    replacement = (100 + np.arange(12, dtype=np.float32)).reshape(3, 4)
    output = np.zeros_like(source)
    descriptor_storage = np.zeros(128, dtype=np.uint8)
    module = numsim.transpile(copied_and_replaced_descriptor_tma_load, cache_dir=tmp_path)

    result = numsim.Engine().run(
        module,
        {
            "source_map": _tensor_map(source),
            "source": source,
            "replacement": replacement,
            "output": output,
            "descriptor_storage": descriptor_storage,
        },
        outputs=("output",),
    )

    np.testing.assert_array_equal(result.outputs["output"], replacement)


def test_typed_payload_copy_ignores_nonzero_descriptor_tail(tmp_path):
    source = np.arange(12, dtype=np.float32).reshape(3, 4)
    replacement = (200 + np.arange(12, dtype=np.float32)).reshape(3, 4)
    output = np.zeros_like(source)
    module = numsim.transpile(copied_and_replaced_descriptor_tma_load, cache_dir=tmp_path)

    result = numsim.Engine().run(
        module,
        {
            "source_map": _tensor_map(source),
            "source": source,
            "replacement": replacement,
            "output": output,
            "descriptor_storage": np.full(128, np.uint8(0xA5), dtype=np.uint8),
        },
        outputs=("output",),
    )

    np.testing.assert_array_equal(result.outputs["output"], replacement)


def test_raw_descriptor_copy_replace_release_and_acquire_drive_raw_tma(tmp_path):
    source = np.arange(12, dtype=np.float32).reshape(3, 4)
    replacement = (100 + np.arange(12, dtype=np.float32)).reshape(3, 4)
    output = np.zeros_like(source)
    module = numsim.transpile(
        raw_copied_and_replaced_descriptor_tma_load,
        cache_dir=tmp_path,
    )

    result = numsim.Engine().run(
        module,
        {
            "source_descriptor_storage": _descriptor_storage(
                storage=np.zeros(16, dtype=np.int64),
                slots={0: _tensor_map(source)},
            ),
            "source": source,
            "replacement": replacement,
            "output": output,
            "descriptor_workspace": np.full(16, -1, dtype=np.int64),
        },
        outputs=("output",),
    )

    np.testing.assert_array_equal(result.outputs["output"], replacement)


def test_raw_descriptor_v4_load_scalar_stores_drive_raw_tma(tmp_path):
    source = np.arange(12, dtype=np.float32).reshape(3, 4)
    output = np.zeros_like(source)
    module = numsim.transpile(raw_descriptor_scalar_store_copy_tma_load, cache_dir=tmp_path)

    result = numsim.Engine().run(
        module,
        {
            "source_descriptor_storage": _descriptor_storage(
                storage=np.zeros(16, dtype=np.int64),
                slots={0: _tensor_map(source)},
            ),
            "output": output,
            "descriptor_workspace": np.zeros(16, dtype=np.int64),
        },
        outputs=("output",),
    )

    np.testing.assert_array_equal(result.outputs["output"], source)


def test_96b_descriptor_copy_ignores_tail_and_drives_raw_tma(tmp_path):
    source = np.arange(12, dtype=np.float32).reshape(3, 4)
    output = np.zeros_like(source)
    module = numsim.transpile(raw_descriptor_96b_copy_tma_load, cache_dir=tmp_path)

    result = numsim.Engine().run(
        module,
        {
            "source_descriptor_storage": _descriptor_storage(
                storage=np.zeros(16, dtype=np.int64),
                slots={0: _tensor_map(source)},
            ),
            "output": output,
            "descriptor_workspace": np.full(16, -1, dtype=np.int64),
        },
        outputs=("output",),
    )

    np.testing.assert_array_equal(result.outputs["output"], source)


def test_32b_raw_descriptor_copy_fails_when_tma_decodes_missing_metadata(tmp_path):
    source = np.arange(12, dtype=np.float32).reshape(3, 4)
    output = np.zeros_like(source)
    module = numsim.transpile(incomplete_raw_descriptor_copy_tma_load, cache_dir=tmp_path)

    with pytest.raises(numsim.NumSimExecutionError, match="image has invalid magic"):
        numsim.Engine().run(
            module,
            {
                "source_descriptor_storage": _descriptor_storage(
                    storage=np.zeros(16, dtype=np.int64),
                    slots={0: _tensor_map(source)},
                ),
                "output": output,
                "descriptor_workspace": np.zeros(16, dtype=np.int64),
            },
            outputs=("output",),
        )


def test_ordinary_u64x4_copy_does_not_require_descriptor_metadata(tmp_path):
    tensor_map_base = np.arange(12, dtype=np.float32).reshape(3, 4)
    source_bytes = _tensor_map(tensor_map_base).copy()
    source_bytes[64] = 1
    source = source_bytes.view(np.int64)
    output = np.zeros_like(source)
    module = numsim.transpile(ordinary_u64x4_copy, cache_dir=tmp_path)

    result = numsim.Engine().run(
        module,
        {
            "source": source,
            "output": output,
        },
        outputs=("output",),
    )

    np.testing.assert_array_equal(result.outputs["output"], source)


@pytest.mark.parametrize(
    ("host_address", "inactive_dimension"),
    [(False, 2), (True, 2), (True, 1)],
    ids=["runtime-inactive-dimension", "host-inactive-dimension", "null-host-address"],
)
def test_ordinary_copy_rejects_noncanonical_tensor_map_candidates(
    tmp_path, host_address, inactive_dimension
):
    from tirx_harness import racecheck, synccheck
    from tirx_harness.numsim.bindings import prepare_bindings

    base = np.arange(12, dtype=np.float32).reshape(3, 4)
    data = np.asarray(_tensor_map(base)).copy()
    # Discovery rejects both noncanonical inactive axes and a null host base.
    # Otherwise descriptor-like scratch bytes must remain ordinary data.
    data[24:28] = np.frombuffer(inactive_dimension.to_bytes(4, "little"), dtype=np.uint8)
    address = 1 << 63 if inactive_dimension != 1 else 0
    data[:8] = np.frombuffer(address.to_bytes(8, "little"), dtype=np.uint8)
    if not host_address:
        data[60] &= np.uint8(0xDF)
    source = data.view(np.int64)
    assert not prepare_bindings({"source": source}).descriptor_allocations

    def inputs():
        return {"source": source.copy(), "output": np.zeros_like(source)}

    for checker in (synccheck, racecheck):
        checker(ordinary_u64x4_copy, inputs()).require_clean()
    module = numsim.transpile(ordinary_u64x4_copy, cache_dir=tmp_path)
    result = numsim.Engine().run(module, inputs(), outputs=("output",))
    assert result.verdict == "clean"
    np.testing.assert_array_equal(result.outputs["output"], source)


def test_raw_descriptor_store_updates_replaced_backing(tmp_path):
    source = np.arange(12, dtype=np.float32).reshape(3, 4) + np.float32(0.5)
    output = np.zeros_like(source)
    output_binding = output
    output_map = numsim.TensorMap(
        base=output_binding,
        global_shape=(4, 3),
        global_strides=(16,),
        box_shape=(4, 3),
        element_strides=(1, 1),
    ).numpy()
    output_map_before = output_map.copy()
    module = numsim.transpile(copied_and_replaced_descriptor_tma_store, cache_dir=tmp_path)

    result = numsim.Engine().run(
        module,
        {
            "output_map": output_map,
            "source": source,
            "output": output_binding,
            "descriptor_storage": np.zeros(128, dtype=np.uint8),
        },
        outputs=("output",),
    )

    np.testing.assert_array_equal(result.outputs["output"], source)
    np.testing.assert_array_equal(output_binding, source)
    np.testing.assert_array_equal(output_map, output_map_before)


def test_host_descriptor_store_updates_declared_base(tmp_path):
    source = np.arange(12, dtype=np.float32).reshape(3, 4) + np.float32(0.25)
    output = np.zeros_like(source)
    output_binding = output
    module = numsim.transpile(host_descriptor_tma_store, cache_dir=tmp_path)

    result = numsim.Engine().run(
        module,
        {
            "source": source,
            "output": output_binding,
            "descriptor_storage": _descriptor_storage(
                storage=np.zeros(16, dtype=np.int64),
                slots={
                    0: _tensor_map(
                        output,
                    )
                },
            ),
        },
        outputs=("output",),
    )

    np.testing.assert_array_equal(result.outputs["output"], source)


def test_stale_descriptor_generation_requires_reacquire(tmp_path):
    source = np.arange(12, dtype=np.float32).reshape(3, 4)
    output = np.zeros_like(source)
    module = numsim.transpile(stale_generation_descriptor_tma_load, cache_dir=tmp_path)

    with pytest.raises(
        numsim.NumSimExecutionError,
        match="latest published generation is not acquired",
    ):
        numsim.Engine().run(
            module,
            {
                "source_map": _tensor_map(source),
                "output": output,
                "descriptor_storage": np.zeros(128, dtype=np.uint8),
            },
            outputs=("output",),
        )


def test_32b_typed_descriptor_copy_fails_when_tma_decodes_missing_metadata(tmp_path):
    source = np.arange(12, dtype=np.float32).reshape(3, 4)
    output = np.zeros_like(source)
    module = numsim.transpile(incomplete_typed_descriptor_copy_tma_load, cache_dir=tmp_path)

    with pytest.raises(numsim.NumSimExecutionError, match="image has invalid magic"):
        numsim.Engine().run(
            module,
            {
                "source_map": _tensor_map(source),
                "output": output,
                "descriptor_storage": np.zeros(128, dtype=np.uint8),
            },
            outputs=("output",),
        )


def test_replace_global_dimension_zero_encodes_documented_2pow32_maximum(tmp_path):
    source = np.arange(12, dtype=np.float32).reshape(3, 4)
    module = numsim.transpile(maximum_dimension_descriptor_update, cache_dir=tmp_path)

    numsim.Engine().run(
        module,
        {
            "source_map": _tensor_map(source),
            "descriptor_storage": np.zeros(128, dtype=np.uint8),
        },
    )


def test_replace_rejects_invalid_dimension_index(tmp_path):
    with pytest.raises(
        UnsupportedTIRxError,
        match="gdn_tensormap_replace_global_dim_5",
    ):
        numsim.transpile(invalid_dimension_index_descriptor_update, cache_dir=tmp_path)


def test_conflicting_host_descriptor_slots_fail_closed(tmp_path):
    source = np.arange(12, dtype=np.float32).reshape(3, 4)
    output = np.zeros_like(source)
    storage = np.zeros(128, dtype=np.uint8)
    module = numsim.transpile(conflicting_host_descriptor_slots, cache_dir=tmp_path)
    descriptor_a = _descriptor_storage(
        storage=storage,
        slots={0: _tensor_map(source)},
    )
    descriptor_b = _descriptor_storage(
        storage=storage,
        slots={0: _tensor_map(source)},
    )

    with pytest.raises(numsim.NumSimExecutionError, match="already bound"):
        numsim.Engine().run(
            module,
            {
                "output": output,
                "descriptor_storage_a": descriptor_a,
                "descriptor_storage_b": descriptor_b,
            },
            outputs=("output",),
        )
