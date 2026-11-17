"""PTX 5.5.3.1: a tiled TMA bounding box must start at a 16-byte-aligned address.

SM107 hardware raises an illegal instruction for a uint32 box whose inner start
coordinate is 1, 2 or -2 (4, 8, -8 bytes) on every descriptor path, while 0, 4
and -4 (0, 16, -16 bytes) copy and zero-fill normally.
"""

import numpy as np
import pytest

import tvm_tirx
from tests.numsim.support.execution import assert_rejected, run_checked
from tirx_harness import numsim
from tvm_tirx.script import tirx as T

REASON = "16-byte aligned"
ROUTES = ("cta", "cluster", "gather4", "store", "reduce", "prefetch")
ALIGNED = (0, 4, -4)
UNALIGNED = (1, 2, 3, -1, -2, -3)


def box_kernel(route, *, raw=False):
    """One rank-2 uint32 TMA issue at runtime coordinates ``(x, y)``."""

    gather = route == "gather4"
    size = 16 if gather else 8
    pointer = "descriptor.ptr_to([0])" if raw else "T.address_of(descriptor)"
    if route in {"cta", "cluster", "gather4"}:
        if gather:
            spelling = "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
            coordinates = "x, y, y + 1, y + 2, y + 3"
        else:
            spelling = f"cp.async.bulk.tensor.2d.shared::{route}.global.tile"
            coordinates = "x, y"
        action = f"""T.ptx.mbarrier.arrive.expect_tx.shared.b64(barrier.ptr_to([0]), {size * 4})
        T.ptx["{spelling}.mbarrier::complete_tx::bytes"](
            shared.ptr_to([0]), {pointer}, {coordinates}, barrier.ptr_to([0]))
        T.cuda.mbarrier_wait(barrier.ptr_to([0]), 0)"""
    elif route in {"store", "reduce"}:
        spelling = (
            "cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group"
            if route == "store"
            else "cp.reduce.async.bulk.tensor.2d.global.shared::cta.add.tile.bulk_group"
        )
        action = f"""T.ptx["{spelling}"]({pointer}, x, y, shared.ptr_to([0]))
        T.ptx.cp.async_.bulk.commit_group()
        T.ptx.cp.async_.bulk.wait_group(0)"""
    elif route == "prefetch":
        action = f'T.ptx["cp.async.bulk.prefetch.tensor.2d.L2.global.tile"]({pointer}, x, y)'
    else:
        raise ValueError(route)
    if raw:
        action = f"T.ptx.fence.proxy.tensormap__generic.acquire.gpu({pointer})\n        {action}"
    descriptor_type = 'T.Tensor((128,), "uint8")' if raw else "T.Ptr(T.TensorMap)"
    return tvm_tirx.script.from_source(
        f"""
@T.function
def kernel(descriptor: {descriptor_type}, x: T.int32, y: T.int32,
           output: T.Tensor(({size},), "uint32")):
    T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=32))
    lane = T.cuda.lane_id()
    shared = T.alloc_shared(({size},), "uint32", align=128)
    barrier = T.alloc_shared((1,), "uint64")
    if lane < {size}:
        shared[lane] = T.uint32(0xEE00) + T.Cast("uint32", lane)
    if lane == 0:
        T.ptx.mbarrier.init.shared.b64(barrier.ptr_to([0]), 1)
    T.ptx.fence.proxy.async_.shared__cta()
    T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.cta_sync()
    if lane == 0:
        {action}
    T.cuda.cta_sync()
    if lane < {size}:
        output[lane] = shared[lane]
""",
        {"T": T},
    )


def box_map(route):
    """Return ``(global, metadata)`` for an (8, 2) or, for gather4, (8, 8) map."""

    rows = 8 if route == "gather4" else 2
    source = (np.arange(8 * rows, dtype=np.uint32) + np.uint32(100)).reshape(rows, 8)
    metadata = dict(
        global_shape=(8, rows),
        global_strides=(32,),
        box_shape=(4, 1 if route == "gather4" else 2),
        element_strides=(1, 1),
    )
    return source, metadata


def box_inputs(route, x, y):
    source, metadata = box_map(route)
    descriptor = numsim.TensorMap(source, **metadata).numpy()
    size = 16 if route == "gather4" else 8
    return {
        "descriptor": descriptor,
        "x": x,
        "y": y,
        "output": np.zeros(size, np.uint32),
    }, source


def expected_box(source, x, rows):
    """Box rows starting at column ``x`` with zero fill outside the tensor."""

    values = []
    for row in rows:
        for column in range(x, x + 4):
            inside = 0 <= row < source.shape[0] and 0 <= column < source.shape[1]
            values.append(source[row, column] if inside else 0)
    return np.asarray(values, np.uint32)


@pytest.mark.parametrize("raw", [False, True])
@pytest.mark.parametrize("route", ["cta", "cluster"])
def test_aligned_tile_load_starts_copy_and_zero_fill(route, raw, tmp_path):
    module = numsim.transpile(box_kernel(route, raw=raw), cache_dir=tmp_path)
    for x in ALIGNED:
        for y in (0, 1, -1):
            inputs, source = box_inputs(route, x, y)
            result = numsim.Engine().run(module, inputs, outputs=("output",))
            np.testing.assert_array_equal(
                result.outputs["output"], expected_box(source, x, (y, y + 1)), err_msg=f"{x, y}"
            )


def test_aligned_gather4_starts_copy_and_zero_fill(tmp_path):
    module = numsim.transpile(box_kernel("gather4"), cache_dir=tmp_path)
    for x in ALIGNED:
        for y in (0, 6, -2):
            inputs, source = box_inputs("gather4", x, y)
            result = numsim.Engine().run(module, inputs, outputs=("output",))
            np.testing.assert_array_equal(
                result.outputs["output"],
                expected_box(source, x, range(y, y + 4)),
                err_msg=f"{x, y}",
            )


@pytest.mark.parametrize("route", ["store", "reduce"])
def test_aligned_tile_store_and_reduce_starts_write_inside_the_tensor(route, tmp_path):
    module = numsim.transpile(box_kernel(route), cache_dir=tmp_path)
    for x in (0, 4):
        for y in (0, 1):
            inputs, source = box_inputs(route, x, y)
            expected = source.copy()
            result = numsim.Engine().run(module, inputs, outputs=("descriptor",))
            contribution = (np.arange(8, dtype=np.uint32) + np.uint32(0xEE00)).reshape(2, 4)
            for row in range(2):
                if y + row < 2:
                    columns = slice(x, x + 4)
                    if route == "store":
                        expected[y + row, columns] = contribution[row]
                    else:
                        expected[y + row, columns] += contribution[row]
            np.testing.assert_array_equal(result.outputs["descriptor"], expected, err_msg=f"{x, y}")


def test_aligned_tile_prefetch_starts_are_accepted():
    for x in ALIGNED:
        inputs, _ = box_inputs("prefetch", x, 0)
        run_checked(box_kernel("prefetch"), inputs, outputs=("output",))


@pytest.mark.parametrize(
    "route,raw", [(route, False) for route in ROUTES] + [("cta", True), ("cluster", True)]
)
@pytest.mark.parametrize("x", UNALIGNED)
def test_unaligned_tile_box_start_is_rejected(route, raw, x, tmp_path):
    if route in {"store", "reduce"} and x < 0:
        pytest.skip("tiled stores already reject negative starting coordinates")
    inputs, _ = box_inputs(route, x, 0)
    assert_rejected(box_kernel(route, raw=raw), inputs, REASON, cache_dir=tmp_path)


# SM107 also traps im2col copies whose uint32 channel start is 1 or 2 and
# copies at 0 and 4, although PTX 5.5.4 states no address rule for im2col.
IM2COL_ROUTES = ("load", "wide", "store", "reduce", "prefetch", "applypriority")


def im2col_box_kernel(route):
    """One im2col issue whose channel start is the runtime scalar ``c``."""

    words = 4 * 16 if route == "wide" else 48
    if route == "load":
        action = """T.ptx.mbarrier.arrive.expect_tx.shared.b64(barrier.ptr_to([0]), 192)
        T.ptx["cp.async.bulk.tensor.3d.shared::cta.global.im2col.mbarrier::complete_tx::bytes"](
            shared.ptr_to([0]), T.address_of(tmap), c, T.int32(-1), T.int32(0),
            barrier.ptr_to([0]), T.uint16(1))
        T.cuda.mbarrier_wait(barrier.ptr_to([0]), 0)"""
    elif route == "wide":
        action = """T.ptx.mbarrier.arrive.expect_tx.shared.b64(barrier.ptr_to([0]), 256)
        T.ptx["cp.async.bulk.tensor.4d.shared::cta.global.im2col::w.mbarrier::complete_tx::bytes"](
            shared.ptr_to([0]), T.address_of(tmap), c, 0, 0, 0,
            barrier.ptr_to([0]), T.uint16(0), T.uint16(0))
        T.cuda.mbarrier_wait(barrier.ptr_to([0]), 0)"""
    elif route in {"store", "reduce"}:
        prefix = "cp.reduce.async.bulk.tensor" if route == "reduce" else "cp.async.bulk.tensor"
        redop = ".add" if route == "reduce" else ""
        action = f"""T.ptx["{prefix}.4d.global.shared::cta{redop}.im2col_no_offs.bulk_group"](
            T.address_of(tmap), c, 1, 0, 0, shared.ptr_to([0]))
        T.ptx.cp.async_.bulk.commit_group()
        T.ptx.cp.async_.bulk.wait_group(0)"""
    elif route == "prefetch":
        action = """T.ptx["cp.async.bulk.prefetch.tensor.3d.L2.global.im2col"](
            T.address_of(tmap), c, T.int32(-1), T.int32(0), T.uint16(1))"""
    elif route == "applypriority":
        action = """T.ptx["applypriority.async.bulk.tensor.3d.global.bulk_group.im2col.L2::evict_normal"](
            T.address_of(tmap), c, T.int32(-1), T.int32(0), T.uint16(1))
        T.ptx.cp.async_.bulk.commit_group()
        T.ptx.cp.async_.bulk.wait_group(0)"""
    else:
        raise ValueError(route)
    return tvm_tirx.script.from_source(
        f"""
@T.function
def kernel(tmap: T.Ptr(T.TensorMap), c: T.int32, out: T.Tensor(({words},), "uint32")):
    T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=32))
    lane = T.cuda.lane_id()
    shared = T.alloc_shared((256,), "uint32", align=1024)
    barrier = T.alloc_shared((1,), "uint64")
    for i in T.serial(8):
        shared[i * 32 + lane] = T.uint32(7)
    if lane == 0:
        T.ptx.mbarrier.init.shared.b64(barrier.ptr_to([0]), 1)
    T.ptx.fence.proxy.async_.shared__cta()
    T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.cta_sync()
    if lane == 0:
        {action}
    T.cuda.cta_sync()
    for i in T.serial({words // 32 + 1}):
        if i * 32 + lane < {words}:
            out[i * 32 + lane] = shared[i * 32 + lane]
""",
        {"T": T},
    )


def im2col_box_map(route):
    """Return ``(source, metadata)`` from shapes the im2col runtime tests use."""

    if route in {"load", "prefetch", "applypriority"}:
        # test_im2col_spatial rank 3; SM107-validated by the alignment probe.
        source = np.arange(64, dtype=np.uint32) + 100
        return source, dict(
            global_shape=(8, 4, 2),
            global_strides=(32, 128),
            box_shape=(4, 12),
            element_strides=(1, 1, 2),
            im2col=numsim.Im2col((-1,), (-1,)),
        )
    if route == "wide":
        # wide_im2col_multiissuer_case shape: 16 channels, 4 pixels, 64B swizzle.
        source = np.arange(16 * 4 * 2 * 2, dtype=np.uint32) + 100
        return source, dict(
            global_shape=(16, 4, 2, 2),
            global_strides=(64, 256, 512),
            box_shape=(16, 4),
            element_strides=(1, 1, 1, 1),
            swizzle="64B",
            im2col=numsim.Im2col((0,), (0,), wide=True),
        )
    # test_im2col_store_and_reduce spatial store shape.
    source = np.full(128, 100, np.uint32)
    return source, dict(
        global_shape=(8, 4, 2, 2),
        global_strides=(32, 128, 256),
        box_shape=(4, 12),
        element_strides=(1, 1, 1, 1),
        im2col=numsim.Im2col((1, 0), (-1, 0)),
    )


def im2col_box_inputs(route, c):
    source, metadata = im2col_box_map(route)
    words = 4 * 16 if route == "wide" else 48
    return {
        "tmap": numsim.TensorMap(source, **metadata).numpy(),
        "c": c,
        "out": np.zeros(words, np.uint32),
    }


def expected_im2col_load(c):
    """test_im2col_spatial's rank-3 first-channel trace shifted by ``c``."""

    first = [100, 108, 116, 124] + [0] * 8
    return np.array(
        [[value + c + k if value else 0 for k in range(4)] for value in first], np.uint32
    ).reshape(-1)


def test_aligned_im2col_channel_starts_copy(tmp_path):
    module = numsim.transpile(im2col_box_kernel("load"), cache_dir=tmp_path)
    for c in (0, 4):
        result = numsim.Engine().run(module, im2col_box_inputs("load", c), outputs=("out",))
        np.testing.assert_array_equal(result.outputs["out"], expected_im2col_load(c))


@pytest.mark.parametrize("route", IM2COL_ROUTES)
def test_aligned_im2col_channel_starts_are_accepted(route):
    for c in (0, 4):
        run_checked(im2col_box_kernel(route), im2col_box_inputs(route, c), outputs=("out",))


@pytest.mark.parametrize("route", IM2COL_ROUTES)
@pytest.mark.parametrize("c", (1, 2))
def test_unaligned_im2col_channel_start_is_rejected(route, c, tmp_path):
    assert_rejected(
        im2col_box_kernel(route), im2col_box_inputs(route, c), REASON, cache_dir=tmp_path
    )
