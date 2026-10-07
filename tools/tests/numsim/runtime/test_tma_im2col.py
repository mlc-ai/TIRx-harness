"""Im2col walks share numeric/checker footprints; row oracles checked on SM100."""

import numpy as np
import pytest
import tvm
from tvm.script import tirx as T

from tests.numsim.support.execution import run_checked
from tirx_harness import numsim


def interleaved_im2col_case(rank, width, *, packed=False, store=False):
    shape = (4,) + (2,) * (rank - 3) + (3, 2)
    strides = tuple(int(np.prod(shape[:i])) * width for i in range(1, rank))
    storage = np.arange(32776, dtype=np.uint32) + 100
    start = (-storage.ctypes.data % 32) // 4
    source = storage[start : start + 32768]
    if packed:
        source[:] = source * np.uint32(0x9E3779B9) ^ np.uint32(0xA5A5A5A5)
    steps = (2,) + (1,) * (rank - 1)
    # Deliberately not the channel-slice width: this field is ignored.
    metadata = numsim.TensorMap(
        source.view(np.uint8) if packed else source,
        shape,
        strides,
        (16, 12),
        steps,
        interleave=f"{width}B",
        swizzle="32B" if width == 32 else None,
        fp4_shared_layout="align8_packed" if packed else None,
        im2col=numsim.Im2col((0,) * (rank - 2), (0,) * (rank - 2)),
    )
    origin = [0] * rank
    origin[-2] = 1
    coords = ", ".join(map(str, origin))
    offsets = ", ".join(["T.uint16(0)"] * (rank - 2))
    words = 12 * width // 4
    index = "((i * 4) ^ (((i * 4 >> 7) & 1) << 4)) // 4" if width == 32 else "i"
    kernel_source = f"""
@T.prim_func
def kernel(tmap: T.TensorMap(), out: T.Buffer(({words},), "uint32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((512,), "uint32", scope="shared", align=1024)
    barrier = T.alloc_buffer((1,), "uint64", scope="shared")
    if lane == 0:
        T.ptx.mbarrier.init.shared.b64(barrier.ptr_to([0]), 1)
    T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.cta_sync()
    if lane == 0:
        T.ptx.mbarrier.arrive.expect_tx.shared.b64(barrier.ptr_to([0]), {words * 4})
        T.ptx["cp.async.bulk.tensor.{rank}d.shared::cta.global.im2col.mbarrier::complete_tx::bytes"](
            shared.ptr_to([0]), T.address_of(tmap), {coords}, barrier.ptr_to([0]), {offsets})
        T.cuda.mbarrier_wait(barrier.ptr_to([0]), 0)
        for i in T.serial({words}):
            out[i] = shared[{index}]
"""
    if store:
        kernel_source = f"""
@T.prim_func
def kernel(tmap: T.TensorMap(), source: T.Buffer(({words},), "uint32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_shared((512,), "uint32", align=1024)
    for i in T.serial({words}):
        if lane == 0:
            shared[{index}] = source[i]
    T.ptx.fence.proxy.async_.shared__cta()
    T.cuda.cta_sync()
    if lane == 0:
        T.ptx["cp.async.bulk.tensor.{rank}d.global.shared::cta.im2col_no_offs.bulk_group"](
            T.address_of(tmap), {coords}, shared.ptr_to([0]))
        T.ptx.cp.async_.bulk.commit_group()
        T.ptx.cp.async_.bulk.wait_group(0)
"""
    kernel = tvm.script.from_source(kernel_source, {"T": T})
    expected = np.zeros(words, np.uint32)
    stored = np.zeros_like(source) if store else None
    # Flatten only the spatial axes, keeping channel slice one fixed.
    for pixel in range(12):
        spatial = pixel
        coordinates = origin.copy()
        for axis in range(rank - 2):
            extent = shape[axis] // steps[axis]
            coordinates[axis] = (spatial % extent) * steps[axis]
            spatial //= extent
        coordinates[-1] = spatial
        if spatial < shape[-1]:
            offset = coordinates[0] * width + sum(c * s for c, s in zip(coordinates[1:], strides))
            expected[pixel * width // 4 : (pixel + 1) * width // 4] = source[
                offset // 4 : (offset + width) // 4
            ]
            if stored is not None:
                stored[offset // 4 : (offset + width) // 4] = source[
                    offset // 4 : (offset + width) // 4
                ]
    if store:
        source.fill(0)
        return (
            kernel,
            {"tmap": metadata.numpy(), "source": expected},
            stored.view(np.uint8) if packed else stored,
            metadata,
        )
    return kernel, {"tmap": metadata.numpy(), "out": np.zeros(words, np.uint32)}, expected, metadata


@pytest.mark.parametrize("rank,width", [(3, 16), (4, 32), (5, 16)])
def test_im2col_interleaved(rank, width, tmp_path):
    for packed in (False, True):
        kernel, args, expected, _ = interleaved_im2col_case(rank, width, packed=packed)
        result = run_checked(kernel, args, cache_dir=tmp_path)
        np.testing.assert_array_equal(result.outputs["out"], expected)
    kernel, args, expected, _ = interleaved_im2col_case(rank, width, packed=True, store=True)
    result = run_checked(kernel, args, cache_dir=tmp_path)
    actual = result.outputs["tmap"].reshape(-1)
    np.testing.assert_array_equal(actual, expected[: actual.size])


def im2col_kernel(rank, pixels, *, override=False, cluster=False):
    coordinates = [0] * rank
    coordinates[1] = -1
    info = [1] + [0] * (rank - 3)
    args = ", ".join(f"T.int32({value})" for value in coordinates)
    extra = ", ".join(f"T.uint16({value})" for value in info)
    suffix = ".override::global_address" if override else ""
    address = ', T.reinterpret("uint64", source.ptr_to([0]))' if override else ""
    return tvm.script.from_source(
        f"""
@T.prim_func
def kernel(tmap: T.TensorMap(), source: T.Buffer((32768,), "uint32"),
           out: T.Buffer(({pixels * 4},), "uint32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    smem = T.alloc_buffer(({max(pixels * 4, 256)},), "uint32", scope="shared", align=1024)
    mbar = T.alloc_buffer((1,), "uint64", scope="shared")
    if lane == 0:
        T.ptx.mbarrier.init.shared.b64(mbar.ptr_to([0]), 1)
    T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.cta_sync()
    if lane == 0:
        T.ptx.mbarrier.arrive.expect_tx.shared.b64(mbar.ptr_to([0]), {pixels * 16})
        T.ptx["cp.async.bulk.tensor.{rank}d.shared::{"cluster" if cluster else "cta"}.global.im2col.mbarrier::complete_tx::bytes{suffix}"](
            smem.ptr_to([0]), T.address_of(tmap){address}, {args}, mbar.ptr_to([0]), {extra})
        T.cuda.mbarrier_wait(mbar.ptr_to([0]), 0)
        for row in T.serial({pixels}):
            for channel in T.serial(4):
                out[row * 4 + channel] = smem[row * 4 + channel]
""",
        {"T": T},
    )


def im2col_inputs(rank, pixels, *, stride=1, batch_stride=1):
    shape = (8,) + (4,) * (rank - 2) + (2,)
    bounds = (-1,) + (0,) * (rank - 3)
    source = np.arange(32768, dtype=np.uint32) + 100
    descriptor = numsim.TensorMap(
        base=source,
        global_shape=shape,
        global_strides=tuple(int(np.prod(shape[:axis])) * 4 for axis in range(1, rank)),
        box_shape=(4, pixels),
        element_strides=(1, stride) + (1,) * (rank - 3) + (batch_stride,),
        im2col=numsim.Im2col(bounds, bounds),
    )
    return {
        "source": source,
        "tmap": descriptor.numpy(),
        "out": np.zeros(pixels * 4, np.uint32),
    }


@pytest.mark.parametrize(
    "rank,stride,batch_stride,override", [(3, 1, 2, False), (4, 2, 1, True), (5, 1, 1, True)]
)
def test_im2col_spatial(rank, stride, batch_stride, override, tmp_path):
    args = im2col_inputs(rank, 12, stride=stride, batch_stride=batch_stride)
    kernel = im2col_kernel(rank, 12, override=override, cluster=rank == 5)
    result = run_checked(kernel, args, cache_dir=tmp_path)
    # Independent first-channel traces from cuTensorMapEncodeIm2col + PTX.
    first = (
        [100, 108, 116, 124] + [0] * 8
        if rank == 3
        else list(range(100, 292, 16))
        if stride == 2
        else list(range(100, 196, 8))
    )
    expected = np.array(
        [[value + c if value else 0 for c in range(4)] for value in first], np.uint32
    )
    np.testing.assert_array_equal(result.outputs["out"].reshape(12, 4), expected)


def im2col_store_kernel(wide=False, reduction=False, override=False):
    mode = "im2col_no_offs::w" if wide else "im2col_no_offs"
    channels = 16 if wide else 4
    prefix = "cp.reduce.async.bulk.tensor" if reduction else "cp.async.bulk.tensor"
    redop = ".add" if reduction else ""
    suffix = ".override::global_address" if override else ""
    address = ', T.reinterpret("uint64", output.ptr_to([0]))' if override else ""
    index = (
        "((row * 64 + channel * 4) ^ (((row * 64 >> 7) & 3) << 4)) // 4"
        if wide
        else "row * 4 + channel"
    )
    return tvm.script.from_source(
        f'''
@T.prim_func
def kernel(tmap: T.TensorMap(), output: T.Buffer((32768,), "uint32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((256,), "uint32", scope="shared", align=1024)
    if lane == 0:
        for row in T.serial(12):
            for channel in T.serial({channels}):
                shared[{index}] = T.Cast("uint32", row * {channels} + channel + 1)
        T.ptx.fence.proxy.async_.shared__cta()
        T.ptx["{prefix}.4d.global.shared::cta{redop}.{mode}.bulk_group{suffix}"](
            T.address_of(tmap){address}, 0, {0 if wide else 1}, 0, 0, shared.ptr_to([0]))
        T.ptx.cp.async_.bulk.commit_group()
        T.ptx.cp.async_.bulk.wait_group(0)
''',
        {"T": T},
    )


@pytest.mark.parametrize(
    "wide,reduction,override",
    [(False, False, False), (False, True, True), (True, False, True), (True, True, False)],
)
def test_im2col_store_and_reduce(wide, reduction, override, tmp_path):
    channels = 16 if wide else 4
    shape = (16 if wide else 8, 4, 2, 2)
    output = np.full(32768, 100, np.uint32)
    tmap = numsim.TensorMap(
        output,
        shape,
        tuple(int(np.prod(shape[:i])) * 4 for i in range(1, 4)),
        (channels, 12),
        (1, 1, 1, 1),
        swizzle="64B" if wide else None,
        im2col=numsim.Im2col((0,) if wide else (1, 0), (0,) if wide else (-1, 0), wide=wide),
    ).numpy()
    args = {"tmap": tmap, "output": output}
    kernel = im2col_store_kernel(wide, reduction, override)
    result = run_checked(kernel, args, cache_dir=tmp_path)
    expected = output.copy()
    locations = (
        [n * 128 + w * 16 for n in range(2) for w in range(4)]
        if wide
        else [n * 64 + h * 32 + w * 8 for n in range(2) for h in range(2) for w in (1, 2)]
    )
    for pixel, start in enumerate(locations):
        expected[start : start + channels] = np.arange(
            pixel * channels + 1, (pixel + 1) * channels + 1, dtype=np.uint32
        ) + (100 if reduction else 0)
    actual = result.outputs["output" if override else "tmap"].reshape(-1)
    np.testing.assert_array_equal(actual, expected[: actual.size])
