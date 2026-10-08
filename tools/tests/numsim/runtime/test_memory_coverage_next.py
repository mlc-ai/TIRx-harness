"""Small observable contracts for the next existing-mechanism memory batch."""

import numpy as np
import pytest
import tvm
from tvm.script import tirx as T

from tirx_harness import numsim, racecheck, synccheck
from tirx_harness.numsim.errors import NumSimExecutionError


@T.prim_func
def uniform_loads(source: T.Buffer((32,), "uint32"), out: T.Buffer((32, 7), "uint32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    values = T.alloc_local((7,), "uint32")
    T.ptx.ldu.global_.u32(values[0], source.ptr_to([3]))
    T.ptx.ldu.v2.u32(values[1], values[2], T.ptx.addr(source.ptr_to([0]), 8))
    T.ptx.ldu.global_.v4.u32(values[3], values[4], values[5], values[6], source.ptr_to([4]))
    for i in T.serial(7):
        out[lane, i] = values[i]


@T.prim_func
def no_complete(count: T.uint32, out: T.Buffer((2,), "uint32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    barrier = T.alloc_buffer((1,), "uint64", scope="shared")
    state = T.alloc_local((1,), "uint64")
    ready = T.alloc_local((1,), "uint32")
    if lane == 0:
        T.ptx.mbarrier.init.shared.b64(barrier.ptr_to([0]), 3)
    T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.cta_sync()
    if lane == 0:
        for phase in T.serial(2):
            T.ptx.mbarrier.arrive.noComplete.shared.b64(
                barrier.ptr_to([0]), T.uint32(99), pred=False
            )
            T.ptx.mbarrier.arrive.noComplete.shared.b64(barrier.ptr_to([0]), count)
            T.ptx.mbarrier.arrive.noComplete.shared.b64(state[0], barrier.ptr_to([0]), T.uint32(1))
            T.ptx.mbarrier.arrive.shared.b64(barrier.ptr_to([0]))
            T.ptx.mbarrier.test_wait.shared.b64(ready[0], barrier.ptr_to([0]), state[0])
            out[phase] = ready[0]


def descriptor_update_kernel(elemtype=11, release=True):
    return tvm.script.from_source(
        f"""
@T.prim_func
def descriptor_update(source_map: T.TensorMap(), descriptor_storage: T.Buffer((16,), "uint64"),
                      out: T.Buffer((8, 8), "float32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((8, 8), "float32", scope="shared", align=256)
    barrier = T.alloc_buffer((1,), "uint64", scope="shared")
    payload = T.decl_buffer((16,), "uint64", data=T.reinterpret("handle", T.address_of(source_map)), scope="param")
    descriptor = descriptor_storage.ptr_to([0])
    if lane == 0:
        for i in T.serial(16):
            descriptor_storage[i] = payload[i]
        T.ptx.tensormap_replace.tile.rank.global_.b1024.b32(descriptor, T.uint32(1))
        T.ptx.tensormap_replace.tile.elemtype.global_.b1024.b32(descriptor, {elemtype})
        T.ptx.tensormap_replace.tile.fill_mode.global_.b1024.b32(descriptor, 1)
        T.ptx.tensormap_replace.tile.swizzle_mode.global_.b1024.b32(descriptor, 1)
        T.ptx.tensormap_replace.tile.global_dim.global_.b1024.b32(descriptor, 0, T.uint32(4))
        {"T.ptx.fence.proxy.tensormap__generic.release.gpu()" if release else "T.evaluate(0)"}
        T.ptx.fence.proxy.tensormap__generic.acquire.gpu(descriptor)
        T.ptx.mbarrier.init.shared.b64(barrier.ptr_to([0]), 1)
    T.ptx.fence.proxy.async_.shared__cta()
    T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.cta_sync()
    if lane == 0:
        T.ptx["cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes"](
            shared.ptr_to([0, 0]), descriptor, 0, 0, barrier.ptr_to([0]))
        T.ptx.mbarrier.arrive.expect_tx.shared.b64(barrier.ptr_to([0]), 256)
        T.cuda.mbarrier_wait(barrier.ptr_to([0]), 0)
    T.cuda.warp_sync()
    for i in T.serial(2):
        out[i * 4 + lane // 8, lane % 8] = shared[i * 4 + lane // 8, lane % 8]
""",
        {"T": T},
    )


def descriptor_arguments(*, paired=False):
    # The input descriptor has rank three and integer elements. All four
    # mutations affect the subsequent rank-two floating-point TMA operation.
    source = (np.arange(64, dtype=np.float32).reshape(8, 8) / 7 + 1).view(np.uint32)
    kwargs = dict(
        global_shape=(8, 8, 1),
        global_strides=(32, 256),
        box_shape=(8, 8, 1),
        element_strides=(1, 1, 1),
    )
    if paired:
        from tests.numsim.microtests.harness import PairedTensorMap

        descriptor = PairedTensorMap(source, **kwargs)
    else:
        descriptor = numsim.TensorMap(source, **kwargs).numpy()
    return {
        "source_map": descriptor,
        "descriptor_storage": np.zeros(16, np.uint64),
        "out": np.zeros((8, 8), np.float32),
    }


def descriptor_expected(elemtype):
    values = np.arange(64, dtype=np.float32).reshape(8, 8) / 7 + 1
    if elemtype == 11:
        # These positive normal values fit f16, which has the TF32 fraction width.
        values = values.astype(np.float16).astype(np.float32)
    values[:, 4:] = np.nan
    # 32B swizzle XORs address bit 4 with bit 7 (four rows per phase).
    values[4:] = values[4:, [4, 5, 6, 7, 0, 1, 2, 3]]
    return values


CASES = ("ldu", "no_complete", "descriptor_f32", "descriptor_tf32")


def memory_case(kind, *, paired=False):
    if kind == "ldu":
        kernel = uniform_loads
        source = np.arange(32, dtype=np.uint32) * 17
        inputs = {"source": source, "out": np.zeros((32, 7), np.uint32)}
        expected = np.tile(source[[3, 2, 3, 4, 5, 6, 7]], (32, 1))
    elif kind == "no_complete":
        kernel, inputs, expected = no_complete, {"count": 1, "out": np.zeros(2, np.uint32)}, [1, 1]
    else:
        elemtype = 7 if kind == "descriptor_f32" else 11
        kernel, inputs, expected = (
            descriptor_update_kernel(elemtype),
            descriptor_arguments(paired=paired),
            descriptor_expected(elemtype),
        )
    return kernel, inputs, expected


@pytest.mark.parametrize("kind", CASES)
def test_next_memory_contracts(kind, tmp_path):
    kernel, inputs, expected = memory_case(kind)
    for checker in (synccheck, racecheck):
        checker(kernel, inputs).require_clean()
    result = numsim.Engine().run(
        numsim.transpile(kernel, cache_dir=tmp_path), inputs, outputs=("out",)
    )
    np.testing.assert_array_equal(result.outputs["out"], expected)


@pytest.mark.parametrize("count", [2, 3])
def test_no_complete_rejects_exhausted_arrivals(count, tmp_path):
    inputs = {"count": count, "out": np.zeros(2, np.uint32)}
    for checker in (synccheck, racecheck):
        report = checker(no_complete, inputs)
        assert report.verdict == "error", report.format()
        assert any("noComplete" in finding.message for finding in report.findings)
    with pytest.raises(NumSimExecutionError, match="noComplete.*pending arrival count"):
        numsim.Engine().run(
            numsim.transpile(no_complete, cache_dir=tmp_path),
            inputs,
        )


@pytest.mark.parametrize("address", ["source.ptr_to([lane])", "source.ptr_to([1])"])
def test_ldu_rejects_nonuniform_or_misaligned_vector(address, tmp_path):
    kernel = tvm.script.from_source(
        f"""
@T.prim_func
def invalid(source: T.Buffer((64,), "uint32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    out = T.alloc_local((2,), "uint32")
    T.ptx.ldu.global_.v2.u32(out[0], out[1], {address})
""",
        {"T": T},
    )
    with pytest.raises(NumSimExecutionError, match="lane-varying|ldu requires 8-byte alignment"):
        numsim.Engine().run(
            numsim.transpile(kernel, cache_dir=tmp_path), {"source": np.zeros(64, np.uint32)}
        )


@pytest.mark.parametrize("elemtype", [8, 12])
def test_tensor_map_ftz_types_preserve_copy_values(elemtype, tmp_path):
    kernel = descriptor_update_kernel(elemtype)
    inputs = descriptor_arguments()
    for checker in (synccheck, racecheck):
        checker(kernel, inputs).require_clean()
    result = numsim.Engine().run(
        numsim.transpile(kernel, cache_dir=tmp_path), inputs, outputs=("out",)
    )
    np.testing.assert_array_equal(result.outputs["out"], descriptor_expected(7 if elemtype == 8 else 11))


def test_tensor_map_update_requires_release(tmp_path):
    with pytest.raises(NumSimExecutionError, match="dirty|release|published"):
        numsim.Engine().run(
            numsim.transpile(descriptor_update_kernel(release=False), cache_dir=tmp_path),
            descriptor_arguments(),
        )
