"""Packed f32x2 ops can write a ``uint64`` view over a ``float32`` pair.

The target PTX DSL uses an explicit 64-bit register/lvalue destination.  A
``float32`` pair therefore exposes the same storage through ``view("uint64")``,
matching canonical kernels that update packed pairs in place.
"""

from __future__ import annotations

import numpy as np

from tirx_harness import numsim
from tirx_harness.numsim.checkers import _run_racecheck as racecheck
from tirx_harness.numsim.checkers import _run_synccheck as synccheck
from tvm.script import tirx as T


@T.prim_func
def packed_f32x2_float32_destination(output: T.Buffer((32, 4, 2), "float32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    pair = T.alloc_buffer((2,), "float32", scope="local")
    pair_u64 = pair.view("uint64")

    lhs: T.let = T.cuda.make_float2(T.cast(lane, "float32"), T.cast(lane + 1, "float32"))
    rhs: T.let = T.cuda.make_float2(T.float32(0.75), T.float32(0.5))
    addend: T.let = T.cuda.make_float2(T.float32(1), T.float32(2))

    T.ptx.fma.rn.ftz.f32x2(pair_u64[0], lhs, rhs, addend)
    output[lane, 0, 0] = pair[0]
    output[lane, 0, 1] = pair[1]

    T.ptx.add.rn.ftz.f32x2(pair_u64[0], lhs, rhs)
    output[lane, 1, 0] = pair[0]
    output[lane, 1, 1] = pair[1]

    T.ptx.sub.rn.ftz.f32x2(pair_u64[0], lhs, rhs)
    output[lane, 2, 0] = pair[0]
    output[lane, 2, 1] = pair[1]

    T.ptx.mul.rn.ftz.f32x2(pair_u64[0], lhs, rhs)
    output[lane, 3, 0] = pair[0]
    output[lane, 3, 1] = pair[1]


@T.prim_func
def packed_f32x2_uint64_destination(output: T.Buffer((32, 4), "uint64")):
    """Byte-identical control whose destination pointee is already 8 bytes."""

    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])

    lhs: T.let = T.cuda.make_float2(T.cast(lane, "float32"), T.cast(lane + 1, "float32"))
    rhs: T.let = T.cuda.make_float2(T.float32(0.75), T.float32(0.5))
    addend: T.let = T.cuda.make_float2(T.float32(1), T.float32(2))

    T.ptx.fma.rn.ftz.f32x2(output[lane, 0], lhs, rhs, addend)
    T.ptx.add.rn.ftz.f32x2(output[lane, 1], lhs, rhs)
    T.ptx.sub.rn.ftz.f32x2(output[lane, 2], lhs, rhs)
    T.ptx.mul.rn.ftz.f32x2(output[lane, 3], lhs, rhs)


@T.prim_func
def packed_f32x2_value_forms(output: T.Buffer((32, 2), "uint64")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    lhs: T.let = T.cuda.make_float2(T.cast(lane, "float32"), T.cast(lane + 1, "float32"))
    rhs: T.let = T.cuda.make_float2(T.float32(0.75), T.float32(0.5))
    addend: T.let = T.cuda.make_float2(T.float32(1), T.float32(2))
    T.ptx.add.f32x2(output[lane, 0], lhs, rhs)
    T.ptx.fma.rn.f32x2(output[lane, 1], lhs, rhs, addend)


def _expected() -> np.ndarray:
    lane = np.arange(32, dtype=np.float32)
    lhs = np.stack([lane, lane + 1], axis=-1)
    rhs = np.array([0.75, 0.5], dtype=np.float32)
    addend = np.array([1.0, 2.0], dtype=np.float32)
    return np.stack([lhs * rhs + addend, lhs + rhs, lhs - rhs, lhs * rhs], axis=1)


def test_packed_f32x2_writes_both_halves_of_a_float32_pair(tmp_path):
    output = np.zeros((32, 4, 2), dtype=np.float32)

    module = numsim.transpile(packed_f32x2_float32_destination, cache_dir=tmp_path)
    result = numsim.Engine().run(module, {"output": output})

    np.testing.assert_array_equal(result.outputs["output"], _expected())


def test_float32_pair_destination_matches_the_uint64_destination(tmp_path):
    float_module = numsim.transpile(packed_f32x2_float32_destination, cache_dir=tmp_path)
    float_result = numsim.Engine().run(
        float_module, {"output": np.zeros((32, 4, 2), dtype=np.float32)}
    )
    uint_module = numsim.transpile(packed_f32x2_uint64_destination, cache_dir=tmp_path)
    uint_result = numsim.Engine().run(uint_module, {"output": np.zeros((32, 4), dtype=np.uint64)})

    np.testing.assert_array_equal(
        float_result.outputs["output"],
        uint_result.outputs["output"].view(np.float32).reshape(32, 4, 2),
    )


def test_packed_f32x2_value_forms_return_both_float_lanes(tmp_path):
    output = np.zeros((32, 2), dtype=np.uint64)
    module = numsim.transpile(packed_f32x2_value_forms, cache_dir=tmp_path)
    result = numsim.Engine().run(module, {"output": output})

    lane = np.arange(32, dtype=np.float32)
    lhs = np.stack([lane, lane + 1], axis=-1)
    rhs = np.array([0.75, 0.5], dtype=np.float32)
    addend = np.array([1.0, 2.0], dtype=np.float32)
    expected = np.stack([lhs + rhs, lhs * rhs + addend], axis=1)
    np.testing.assert_array_equal(
        result.outputs["output"].view(np.float32).reshape(32, 2, 2), expected
    )


def test_checkers_accept_the_float32_pair_destination(tmp_path):
    inputs = {"output": np.zeros((32, 4, 2), dtype=np.float32)}

    sync = synccheck(
        packed_f32x2_float32_destination,
        inputs=inputs,
        cache_dir=tmp_path,
        max_workers=1,
    )
    sync.require_clean()

    race = racecheck(packed_f32x2_float32_destination, inputs=inputs)
    race.require_clean()


@T.prim_func
def packed_store_races_with_second_element(output: T.Buffer((2,), "float32")):
    """The packed store spans smem[0..2); warp 1 concurrently writes smem[1]."""

    T.device_entry()
    warp = T.warp_id([2])
    lane = T.lane_id([32])
    smem = T.alloc_buffer((2,), "float32", scope="shared")
    smem_pair = smem.view("uint64")

    lhs: T.let = T.cuda.make_float2(T.float32(1), T.float32(2))
    rhs: T.let = T.cuda.make_float2(T.float32(3), T.float32(4))

    if warp == 0 and lane == 0:
        T.ptx.add.rn.ftz.f32x2(smem_pair[0], lhs, rhs)
    if warp == 1 and lane == 0:
        smem[1] = T.float32(7)
    T.cuda.cta_sync()
    if warp == 0 and lane == 0:
        output[0] = smem[0]
        output[1] = smem[1]


def test_packed_store_footprint_covers_the_second_element():
    """Positive control: the recorded footprint must be the full 8 bytes.

    A packed f32x2 write through the uint64 view covers two float32 elements.
    Missing the aliased second half would incorrectly report this real
    write-write race as clean.
    """

    report = racecheck(
        packed_store_races_with_second_element,
        inputs={"output": np.zeros(2, dtype=np.float32)},
    )

    assert report.verdict == "error", report.verdict
    assert [f.details["access_pair"] for f in report.findings] == ["write_write"]
