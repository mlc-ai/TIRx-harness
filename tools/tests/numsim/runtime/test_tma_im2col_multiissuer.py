"""Im2col keeps per-thread coordinates, offsets, halos and completion counts.

Poll with raw PTX, not cuda.mbarrier_wait's uniform-branch helper: different
lane barriers may complete at different times.
"""

import numpy as np
import pytest
import tvm
from tvm.script import tirx as T

from tests.numsim.support.execution import run_checked
from tirx_harness import numsim, racecheck


def im2col_multiissuer_case(rank, sparse=False, *, wait=True):
    multicast = rank == 5
    ctas = 2 if multicast else 1
    active = "lane % 3 == 0" if sparse else "True"
    selected = "(lane % 3 == 0 or cta == lane % 2)" if multicast else "True"
    coordinates = ["lane % 2 * 4", "lane // 2 % 2", *(["0"] * (rank - 3)), "lane"]
    offsets = [f"T.uint16((lane >> {axis}) & 1)" for axis in range(rank - 2)]
    extra = ", T.uint16(T.Select(lane % 3 == 0, 3, 1 << (lane % 2)))" if multicast else ""
    instruction = (
        f"cp.async.bulk.tensor.{rank}d.shared::{'cta' if rank == 3 else 'cluster'}"
        ".global.im2col.mbarrier::complete_tx::bytes"
        + (".multicast::cluster" if multicast else "")
    )
    kernel = tvm.script.from_source(
        f"""
@T.prim_func
def kernel(tmap: T.TensorMap(), output: T.Buffer(({ctas}, 32, 32), "uint32")):
    T.device_entry()
    _cluster = T.cluster_id([1])
    cta = T.cta_id_in_cluster([{ctas}])
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((1024,), "uint32", scope="shared", align=128)
    barriers = T.alloc_buffer((32,), "uint64", scope="shared")
    ready = T.alloc_local((1,), "uint32")
    for i in T.serial(32):
        shared[lane * 32 + i] = T.uint32(7)
    T.ptx.mbarrier.init.shared.b64(barriers.ptr_to([lane]), 1)
    T.ptx.fence.proxy.async_.shared__cta()
    T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.cluster_sync()
    if ({active}) and ({selected}):
        T.ptx.mbarrier.arrive.expect_tx.shared.b64(barriers.ptr_to([lane]), 32)
    T.cuda.cluster_sync()
    if cta == 0:
        T.ptx["{instruction}"](shared.ptr_to([lane * 32]), T.address_of(tmap),
            {", ".join(coordinates)}, barriers.ptr_to([lane]), {", ".join(offsets)}{extra}, pred={active})
    if {wait} and ({active}) and ({selected}):
        ready[0] = T.uint32(0)
        while ready[0] == 0:
            T.ptx.mbarrier.try_wait.parity.shared.b64(
                ready[0], barriers.ptr_to([lane]), T.uint32(0), T.uint32(1))
    T.cuda.cluster_sync()
    for i in T.serial(32):
        output[cta, lane, i] = shared[lane * 32 + i]
""",
        {"T": T},
    )
    shape = (8,) + (4,) * (rank - 2) + (32,)
    source = np.arange(int(np.prod(shape)), dtype=np.uint32) + 100
    element_strides = [int(np.prod(shape[:axis])) for axis in range(rank)]
    metadata = dict(
        global_shape=shape,
        global_strides=tuple(stride * 4 for stride in element_strides[1:]),
        box_shape=(4, 2),
        element_strides=(1,) * rank,
        im2col=numsim.Im2col((0,) * (rank - 2), (0,) * (rank - 2)),
    )
    expected = np.full((ctas, 32, 32), 7, np.uint32)
    for lane in range(32):
        if sparse and lane % 3:
            continue
        # Two adjacent W pixels, with independent C/N coordinates and filter
        # offsets. No carry across spatial dimensions in this input domain.
        origin = [lane % 2 * 4, lane // 2 % 2] + [0] * (rank - 3) + [lane]
        for axis in range(rank - 2):
            origin[axis + 1] += (lane >> axis) & 1
        base = sum(coord * stride for coord, stride in zip(origin, element_strides))
        values = np.concatenate([source[base + row * 8 : base + row * 8 + 4] for row in range(2)])
        for cta in range(ctas):
            if not multicast or lane % 3 == 0 or cta == lane % 2:
                expected[cta, lane, :8] = values
    return kernel, source, metadata, expected


@pytest.mark.parametrize("rank", [3, 4, 5])
def test_im2col_per_thread_coordinates_and_offsets(rank, tmp_path):
    for sparse in (False, True):
        kernel, source, metadata, expected = im2col_multiissuer_case(rank, sparse)
        args = {"tmap": numsim.TensorMap(source, **metadata).numpy(), "output": np.zeros_like(expected)}
        result = run_checked(kernel, args, cache_dir=tmp_path)
        np.testing.assert_array_equal(result.outputs["output"], expected)


def test_im2col_multiissuer_requires_completion_wait():
    kernel, source, metadata, expected = im2col_multiissuer_case(3, wait=False)
    report = racecheck(kernel, {
        "tmap": numsim.TensorMap(source, **metadata).numpy(), "output": np.zeros_like(expected),
    })
    assert report.verdict == "error", report.format()
    assert any(f.status == "error" and f.details["access_pair"] in {"write_read", "read_write"} for f in report.findings), report.format()


def wide_im2col_multiissuer_case(w128, sparse=False):
    pixels, multiplier, stride = (128, 4, 2560) if w128 else (5, 1, 256)
    max_rows = pixels + 2 * multiplier
    active = "lane < 4 and lane != 1" if sparse else "lane < 4"
    mode = "im2col::w::128" if w128 else "im2col::w"
    kernel = tvm.script.from_source(
        f"""
@T.prim_func
def kernel(tmap: T.TensorMap(), output: T.Buffer((4, {max_rows}, 16), "uint32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_buffer(({4 * stride},), "uint32", scope="shared", align=1024)
    barriers = T.alloc_buffer((4,), "uint64", scope="shared")
    ready = T.alloc_local((1,), "uint32")
    if lane < 4:
        for i in T.serial({stride}):
            shared[lane * {stride} + i] = T.uint32(7)
        T.ptx.mbarrier.init.shared.b64(barriers.ptr_to([lane]), 1)
    T.ptx.fence.proxy.async_.shared__cta()
    T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.cta_sync()
    if {active}:
        T.ptx.mbarrier.arrive.expect_tx.shared.b64(
            barriers.ptr_to([lane]), T.uint32(({pixels} + lane % 3 * {multiplier}) * 64))
        T.ptx["cp.async.bulk.tensor.4d.shared::cta.global.{mode}.mbarrier::complete_tx::bytes"](
            shared.ptr_to([lane * {stride}]), T.address_of(tmap),
            0, 0, lane % 2, lane * 40, barriers.ptr_to([lane]), T.uint16(lane % 3), T.uint16(lane % 2))
        ready[0] = T.uint32(0)
        while ready[0] == 0:
            T.ptx.mbarrier.try_wait.parity.shared.b64(
                ready[0], barriers.ptr_to([lane]), T.uint32(0), T.uint32(1))
    if lane < 4:
        for row in T.serial({max_rows}):
            for channel in T.serial(16):
                output[lane, row, channel] = shared[
                    lane * {stride} + ((row * 64 + channel * 4) ^ (((row * 64 >> 7) & 3) << 4)) // 4]
""",
        {"T": T},
    )
    source = np.arange(16 * 4 * 2 * 160, dtype=np.uint32) + 100
    metadata = dict(
        global_shape=(16, 4, 2, 160), global_strides=(64, 256, 512),
        box_shape=(16, pixels), element_strides=(1, 1, 1, 1), swizzle="64B",
        im2col=numsim.Im2col((0,), (0,), wide=True),
    )
    expected = np.full((4, max_rows, 16), 7, np.uint32)
    for lane in range(4):
        if sparse and lane == 1:
            continue
        halo = lane % 3
        # Fixed-128 packs four post-32-pixel halos in a separate plane;
        # ordinary wide appends the halo after its requested pixel walk.
        positions = (
            list(range(128)) + [32 * (group + 1) + h for h in range(halo) for group in range(4)]
            if w128 else list(range(pixels + halo))
        )
        for row, position in enumerate(positions):
            w, n = position % 4 + lane % 2, position // 4 + lane * 40
            start = (w + 4 * (lane % 2 + 2 * n)) * 16
            expected[lane, row] = source[start : start + 16] if w < 4 else 0
    return kernel, source, metadata, expected


@pytest.mark.parametrize("w128", [False, True])
def test_wide_im2col_per_thread_halo_offset_and_completion(w128, tmp_path):
    for sparse in (False, True):
        kernel, source, metadata, expected = wide_im2col_multiissuer_case(w128, sparse)
        args = {"tmap": numsim.TensorMap(source, **metadata).numpy(), "output": np.zeros_like(expected)}
        result = run_checked(kernel, args, cache_dir=tmp_path)
        np.testing.assert_array_equal(result.outputs["output"], expected)
        if not sparse:
            # Preserve the independent single-issuer row oracle at this lane's
            # batch origin, without compiling a second kernel for the same walk.
            first = (
                [100 + (i // 4) * 128 + (i % 4) * 16 for i in range(128)]
                + [1124, 2148, 3172, 4196, 1140, 2164, 3188, 4212]
                if w128 else [116, 132, 148, 0, 244, 260]
            )
            lane, base = (2, 10240) if w128 else (1, 5184)
            rows = np.array(
                [[value + base + c if value else 0 for c in range(16)] for value in first],
                np.uint32,
            )
            np.testing.assert_array_equal(result.outputs["output"][lane, :len(first)], rows)
