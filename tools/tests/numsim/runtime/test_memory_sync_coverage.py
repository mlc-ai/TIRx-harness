"""Representative semantics for existing-mechanism PTX coverage additions."""

import numpy as np
import pytest
from tvm.script import tirx as T

from tirx_harness import numsim, racecheck, synccheck


@T.prim_func
def scalar_atomic_sinks(cell: T.Buffer((1,), "uint32"), out: T.Buffer((3,), "uint32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    if lane == 0:
        old = T.alloc_local((1,), "uint32")
        T.ptx.atom.relaxed.gpu.global_.exch.b32(old[0], cell.ptr_to([0]), T.uint32(7))
        out[0] = old[0]
        T.ptx.atom.relaxed.gpu.global_.cas.b32(cell.ptr_to([0]), T.uint32(7), T.uint32(9))
        T.ptx.atom.relaxed.gpu.global_.cas.b32(cell.ptr_to([0]), T.uint32(7), T.uint32(99))
        out[1] = cell[0]
        T.ptx.atom.relaxed.gpu.global_.exch.b32(cell.ptr_to([0]), T.uint32(11))
        T.ptx.atom.relaxed.gpu.global_.add.u32(cell.ptr_to([0]), T.uint32(2))
        out[2] = cell[0]


@T.prim_func
def half_atomic_sinks(cell: T.Buffer((1,), "float16"), out: T.Buffer((1,), "uint16")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    if lane == 0:
        old = T.alloc_local((1,), "uint16")
        T.ptx.atom.relaxed.gpu.global_.add.noftz.f16(old[0], cell.ptr_to([0]), T.uint16(0x3800))
        out[0] = old[0]
        T.ptx.atom.relaxed.gpu.global_.add.noftz.f16(cell.ptr_to([0]), T.uint16(0x3400))


@T.prim_func
def copy_wait_all(source: T.Buffer((32,), "uint32"), out: T.Buffer((32,), "uint32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((32,), "uint32", scope="shared")
    T.ptx.cp.async_.ca.shared.global_(shared.ptr_to([lane]), source.ptr_to([lane]), 4)
    # No explicit commit: wait_all must include these outstanding copies.
    T.ptx.cp.async_.wait_all()
    out[lane] = shared[lane]
    T.ptx.cp.async_.wait_all()


@T.prim_func
def barrier_state_queries(source: T.Buffer((4,), "uint32"), out: T.Buffer((4,), "uint32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    barriers = T.alloc_buffer((2,), "uint64", scope="shared")
    shared = T.alloc_buffer((4,), "uint32", scope="shared")
    state = T.alloc_local((1,), "uint64")
    ready = T.alloc_local((1,), "uint32")
    if lane == 0:
        T.ptx.mbarrier.init.shared.b64(barriers.ptr_to([0]), 1)
        T.ptx.mbarrier.init.shared.b64(barriers.ptr_to([1]), 1)
    T.ptx.fence.proxy.async_.shared__cta()
    T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.cta_sync()
    if lane == 0:
        T.ptx.mbarrier.arrive.shared.b64(state[0], barriers.ptr_to([0]))
        T.ptx.mbarrier.test_wait.shared.b64(ready[0], barriers.ptr_to([0]), state[0])
        out[0] = ready[0]
        T.ptx.mbarrier.arrive.expect_tx.shared.b64(state[0], barriers.ptr_to([1]), 16)
        T.ptx.mbarrier.test_wait.shared.b64(ready[0], barriers.ptr_to([1]), state[0])
        out[1] = ready[0]
        T.ptx["cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes"](
            shared.ptr_to([0]), source.ptr_to([0]), 16, barriers.ptr_to([1])
        )
        while ready[0] == 0:
            T.ptx.mbarrier.try_wait.shared.b64(ready[0], barriers.ptr_to([1]), state[0], 10)
        out[2] = ready[0]
        out[3] = shared[0]


@T.prim_func
def vector_atomic_sinks(
    half: T.Buffer((2,), "float16"),
    floats: T.Buffer((2,), "float32"),
    out: T.Buffer((2,), "uint16"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    if lane == 0:
        old = T.alloc_local((2,), "uint16")
        T.ptx.atom.relaxed.gpu.global_.add.noftz.v2.f16(
            old[0], old[1], half.ptr_to([0]), T.uint16(0x3800), T.uint16(0x3400)
        )
        out[0] = old[0]
        out[1] = old[1]
        T.ptx.atom.relaxed.gpu.global_.add.noftz.v2.f16(
            half.ptr_to([0]), T.uint16(0x3800), T.uint16(0x3400)
        )
        T.ptx.red.relaxed.gpu.global_.add.noftz.v2.f16(
            half.ptr_to([0]), T.uint16(0x3800), T.uint16(0x3400)
        )
        T.ptx.atom.relaxed.gpu.global_.add.v2.f32(
            floats.ptr_to([0]), T.float32(0.25), T.float32(0.5)
        )
        T.ptx.red.relaxed.gpu.global_.add.v2.f32(
            floats.ptr_to([0]), T.float32(0.25), T.float32(0.5)
        )


@T.prim_func
def address_queries(out: T.Buffer((2, 10), "uint32")):
    T.device_entry()
    _cluster = T.cluster_id([1])
    cta = T.cta_id_in_cluster([2])
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((1,), "uint32", scope="shared")
    raw = T.alloc_local((1,), "uint64")
    mapped = T.alloc_local((1,), "uint64")
    generic = T.alloc_local((1,), "uint64")
    result = T.alloc_local((1,), "uint32")
    if lane == 0:
        T.ptx.cvta.to.shared__cluster.u64(raw[0], shared.ptr_to([0]))
        T.ptx.mapa.shared__cluster.u64(mapped[0], raw[0], T.uint32(1 - cta))
        T.ptx.getctarank.shared__cluster.u64(result[0], mapped[0])
        out[cta, 0] = result[0]
        generic[0] = T.reinterpret("uint64", shared.ptr_to([0]))
        T.ptx.mapa.u64(mapped[0], generic[0], T.uint32(1 - cta))
        T.ptx.getctarank.u64(result[0], mapped[0])
        out[cta, 1] = result[0]
        T.ptx.isspacep.shared(result[0], shared.ptr_to([0]))
        out[cta, 2] = result[0]
        T.ptx.isspacep.global_(result[0], shared.ptr_to([0]))
        out[cta, 3] = result[0]
        T.ptx.isspacep.shared__cluster(result[0], T.reinterpret("handle", mapped[0]))
        out[cta, 4] = result[0]
        T.ptx.isspacep.shared__cta(result[0], T.reinterpret("handle", mapped[0]))
        out[cta, 5] = result[0]
        T.ptx.mapa.u64(mapped[0], generic[0], T.uint32(cta))
        T.ptx.isspacep.shared__cta(result[0], T.reinterpret("handle", mapped[0]))
        out[cta, 6] = result[0]
        T.ptx.isspacep.global_(result[0], out.ptr_to([cta, 0]))
        out[cta, 7] = result[0]
        T.ptx.isspacep.local(result[0], result.ptr_to([0]))
        out[cta, 8] = result[0]
        T.ptx.cvta.shared__cluster.u64(generic[0], raw[0])
        T.ptx.isspacep.shared__cluster(result[0], T.reinterpret("handle", generic[0]))
        out[cta, 9] = result[0]


@T.prim_func
def scalar_async_store(out: T.Buffer((4,), "uint32")):
    T.device_entry()
    _cluster = T.cluster_id([1])
    cta = T.cta_id_in_cluster([2])
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((4,), "uint32", scope="shared", align=16)
    barrier = T.alloc_buffer((1,), "uint64", scope="shared")
    if cta == 1 and lane == 0:
        shared[1] = T.uint32(99)
        T.ptx.mbarrier.init.shared.b64(barrier.ptr_to([0]), 1)
    T.ptx.fence.proxy.async_.shared__cluster()
    T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.cluster_sync()
    if cta == 0 and lane == 0:
        remote = T.alloc_local((1,), "uint32")
        remote_barrier = T.alloc_local((1,), "uint32")
        T.ptx.mapa.shared__cluster.u32(
            remote[0], T.cuda.cvta_generic_to_shared(shared.ptr_to([0])), 1
        )
        T.ptx.mapa.shared__cluster.u32(
            remote_barrier[0], T.cuda.cvta_generic_to_shared(barrier.ptr_to([0])), 1
        )
        T.ptx.mbarrier.arrive.expect_tx.shared__cluster.b64(remote_barrier[0], 12)
        T.ptx.st_async.shared__cluster.mbarrier__complete_tx__bytes.u32(
            remote[0], T.uint32(7), remote_barrier[0]
        )
        T.ptx.mapa.shared__cluster.u32(
            remote[0], T.cuda.cvta_generic_to_shared(shared.ptr_to([2])), 1
        )
        T.ptx.st_async.shared__cluster.mbarrier__complete_tx__bytes.u64(
            remote[0], T.uint64(0x1122334455667788), remote_barrier[0]
        )
    if cta == 1 and lane == 0:
        T.cuda.mbarrier_wait(barrier.ptr_to([0]), 0)
    T.cuda.cluster_sync()
    if cta == 1 and lane < 4:
        out[lane] = shared[lane]


CASES = (
    (
        scalar_atomic_sinks,
        {"cell": np.array([3], dtype=np.uint32), "out": np.zeros(3, dtype=np.uint32)},
        {"cell": [13], "out": [3, 9, 13]},
    ),
    (
        half_atomic_sinks,
        {"cell": np.array([1], dtype=np.float16), "out": np.zeros(1, dtype=np.uint16)},
        {"cell": [1.75], "out": [0x3C00]},
    ),
    (
        copy_wait_all,
        {"source": np.arange(32, dtype=np.uint32), "out": np.zeros(32, dtype=np.uint32)},
        {"out": np.arange(32)},
    ),
    (
        barrier_state_queries,
        {"source": np.array([7, 8, 9, 10], dtype=np.uint32), "out": np.zeros(4, dtype=np.uint32)},
        {"out": [1, 0, 1, 7]},
    ),
    (
        vector_atomic_sinks,
        {
            "half": np.array([1, 2], dtype=np.float16),
            "floats": np.array([3, 4], dtype=np.float32),
            "out": np.zeros(2, dtype=np.uint16),
        },
        {"half": [2.5, 2.75], "floats": [3.5, 5], "out": [0x3C00, 0x4000]},
    ),
    (
        address_queries,
        {"out": np.zeros((2, 10), dtype=np.uint32)},
        {"out": [[1, 1, 1, 0, 1, 0, 1, 1, 1, 1], [0, 0, 1, 0, 1, 0, 1, 1, 1, 1]]},
    ),
    (
        scalar_async_store,
        {"out": np.zeros(4, dtype=np.uint32)},
        {"out": [7, 99, 0x55667788, 0x11223344]},
    ),
)


@pytest.mark.parametrize(
    "kernel,inputs,expected", CASES, ids=lambda value: getattr(value, "__name__", None)
)
def test_memory_sync_extensions(kernel, inputs, expected, tmp_path):
    for checker in (synccheck, racecheck):
        checker(kernel, {name: value.copy() for name, value in inputs.items()}).require_clean()
    result = numsim.Engine().run(
        numsim.transpile(kernel, cache_dir=tmp_path),
        {name: value.copy() for name, value in inputs.items()},
        outputs=tuple(expected),
    )
    for name, value in expected.items():
        np.testing.assert_array_equal(result.outputs[name], value)


def test_half_vector_predicate_alias_is_captured_before_either_result_store(tmp_path):
    @T.prim_func
    def kernel(enabled: T.uint16, cell: T.Buffer((2,), "float16"), out: T.Buffer((2,), "uint16")):
        T.device_entry()
        _warp = T.warp_id([1])
        lane = T.lane_id([32])
        if lane == 0:
            old = T.alloc_local((2,), "uint16")
            old[0] = enabled
            old[1] = T.uint16(0xDEAD)
            T.ptx.atom.relaxed.gpu.global_.add.noftz.v2.f16(
                old[0], old[1], cell.ptr_to([0]), T.uint16(0x3800), T.uint16(0x3400), pred=old[0] != 0
            )
            out[0] = old[0]
            out[1] = old[1]

    module = numsim.transpile(kernel, cache_dir=tmp_path)
    for enabled in (0, 1):
        inputs = {
            "enabled": enabled,
            "cell": np.array([0, 2], np.float16),
            "out": np.zeros(2, np.uint16),
        }
        for checker in (synccheck, racecheck):
            checker(
                kernel,
                {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in inputs.items()},
            ).require_clean()
        result = numsim.Engine().run(module, inputs)
        # When enabled, the first result clears the predicate's source to zero;
        # that must not suppress the second result or its memory update.
        np.testing.assert_array_equal(result.outputs["out"], [0, 0x4000 if enabled else 0xDEAD])
        np.testing.assert_array_equal(result.outputs["cell"], [0.5, 2.25] if enabled else [0, 2])


def test_half_atomic_old_value_preserves_signed_zero_subnormal_and_nan_bits(tmp_path):
    module = numsim.transpile(half_atomic_sinks, cache_dir=tmp_path)
    for bits in (0x8000, 0x0001, 0x7C01, 0x7E55):
        result = numsim.Engine().run(
            module,
            {
                "cell": np.array([bits], dtype=np.uint16).view(np.float16),
                "out": np.zeros(1, dtype=np.uint16),
            },
            outputs=("out",),
        )
        np.testing.assert_array_equal(result.outputs["out"], [bits])
