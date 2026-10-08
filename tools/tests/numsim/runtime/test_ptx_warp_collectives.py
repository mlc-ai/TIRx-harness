from __future__ import annotations

import numpy as np
import pytest

from tirx_harness import numsim
from tirx_harness.numsim.checkers import _run_racecheck, _run_synccheck
from tests.numsim.support.manifest import call_op_names
from tvm.script import tirx as T


@T.prim_func
def raw_ptx_warp_collectives(
    source: T.Buffer((32,), "float32"),
    output: T.Buffer((32, 6), "uint32"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shuffled = T.alloc_local((1,), "uint32")
    shuffled_p = T.alloc_local((1,), "uint32")
    in_range = T.alloc_local((1,), "uint32")
    elected_lane = T.alloc_local((1,), "uint32")
    is_elected = T.alloc_local((1,), "uint32")
    full = T.uint32(0xFFFFFFFF)

    T.ptx.shfl_sync.idx.b32(
        shuffled[0],
        source[lane],
        T.cast((lane * 7 + 3) % 32, "uint32"),
        T.uint32(31),
        full,
    )
    output[lane, 0] = shuffled[0]
    T.ptx.shfl_sync.bfly.b32(shuffled[0], source[lane], T.uint32(1), T.uint32(31), full)
    output[lane, 1] = shuffled[0]
    T.ptx.shfl_sync.down.b32(
        shuffled_p[0],
        in_range[0],
        source[lane],
        T.uint32(2),
        T.uint32(31),
        full,
    )
    T.ptx.elect_sync(elected_lane[0], is_elected[0], full)
    output[lane, 2] = shuffled_p[0]
    output[lane, 3] = in_range[0]
    output[lane, 4] = elected_lane[0]
    output[lane, 5] = is_elected[0]


@T.prim_func
def raw_ptx_shuffle_sparse_sources(
    selector: T.Buffer((32,), "uint32"),
    output: T.Buffer((32,), "uint32"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    source = T.alloc_local((1,), "float32")
    shuffled = T.alloc_local((1,), "uint32")

    if lane % 4 == 2:
        source[0] = T.cast(lane + 100, "float32")
    T.ptx.shfl_sync.idx.b32(
        shuffled[0],
        T.reinterpret("uint32", source[0]),
        selector[lane],
        T.uint32(31),
        T.uint32(0xFFFFFFFF),
    )
    output[lane] = shuffled[0]


@T.prim_func
def raw_ptx_f32_reductions(
    source: T.Buffer((32,), "float32"),
    zeros: T.Buffer((32,), "float32"),
    output: T.Buffer((32, 6), "float32"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    full = T.uint32(0xFFFFFFFF)
    T.ptx.redux_sync.max.NaN.f32(output[lane, 0], source[lane], full)
    T.ptx.redux_sync.min.NaN.f32(output[lane, 1], source[lane], full)
    T.ptx.redux_sync.max.f32(output[lane, 2], source[lane], full)
    T.ptx.redux_sync.min.f32(output[lane, 3], source[lane], full)
    T.ptx.redux_sync.max.f32(output[lane, 4], zeros[lane], full)
    T.ptx.redux_sync.min.f32(output[lane, 5], zeros[lane], full)


@T.prim_func
def raw_ptx_vote_and_integer_reductions(
    predicates: T.Buffer((32,), "bool"),
    values: T.Buffer((32,), "uint32"),
    output: T.Buffer((32, 6), "uint32"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    full = T.uint32(0xFFFFFFFF)
    T.ptx.vote_sync.any.pred(output[lane, 0], predicates[lane], full)
    T.ptx.vote_sync.all.pred(output[lane, 1], predicates[lane], full)
    T.ptx.vote_sync.uni.pred(output[lane, 2], predicates[lane], full)
    T.ptx.vote_sync.ballot.b32(output[lane, 3], predicates[lane], full)
    T.ptx.redux_sync.add.u32(output[lane, 4], values[lane], full)
    T.ptx.redux_sync.min.u32(output[lane, 5], values[lane], full)


@T.prim_func
def raw_ptx_vote_predicate_carrier(
    predicates: T.Buffer((32,), "uint32"),
    output_u32: T.Buffer((32,), "uint32"),
    output_i32: T.Buffer((32,), "int32"),
    output_f32: T.Buffer((32,), "float32"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    T.ptx.vote_sync.ballot.b32(
        output_u32[lane],
        T.ptx.pred(predicates[lane]),
        T.uint32(0xFFFFFFFF),
    )
    T.ptx.vote_sync.ballot.b32(
        output_i32[lane],
        T.ptx.pred(predicates[lane]),
        T.uint32(0xFFFFFFFF),
    )
    T.ptx.vote_sync.ballot.b32(
        output_f32[lane],
        T.ptx.pred(predicates[lane]),
        T.uint32(0xFFFFFFFF),
    )


@T.prim_func
def raw_ptx_match_redux_and_activemask(
    source32: T.Buffer((32,), "uint32"),
    source64: T.Buffer((32,), "uint64"),
    partial_mask: T.Buffer((1,), "uint32"),
    output: T.Buffer((32, 13), "uint32"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    full = T.uint32(0xFFFFFFFF)
    active = T.alloc_local((1,), "uint32")
    all_equal = T.alloc_local((1,), "uint32")

    if lane < 8:
        T.ptx.activemask.b32(active[0])
        output[lane, 0] = active[0]
        T.ptx.match.any.sync.b32(output[lane, 11], source32[lane], partial_mask[0])
        T.ptx.redux_sync.xor.b32(output[lane, 12], source32[lane], partial_mask[0])
    T.ptx.match.any.sync.b32(output[lane, 1], source32[lane], full)
    T.ptx.match.any.sync.b64(output[lane, 2], source64[lane], full)
    T.ptx.match.all.sync.b32(output[lane, 3], source32[lane], full)
    T.ptx.match.all.sync.b32(output[lane, 4], all_equal[0], source32[lane], full)
    output[lane, 5] = all_equal[0]
    T.ptx.match.all.sync.b64(output[lane, 6], all_equal[0], T.uint64(0x0123456789ABCDEF), full)
    output[lane, 7] = all_equal[0]
    T.ptx.redux_sync.and_.b32(output[lane, 8], source32[lane], full)
    T.ptx.redux_sync.or_.b32(output[lane, 9], source32[lane], full)
    T.ptx.redux_sync.xor.b32(output[lane, 10], source32[lane], full)


@T.prim_func
def raw_ptx_match_float_carriers(
    source32: T.Buffer((32,), "float32"),
    source64: T.Buffer((32,), "float64"),
    output32: T.Buffer((32,), "int32"),
    output64: T.Buffer((32,), "uint32"),
    active: T.Buffer((32,), "float32"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    mask = T.alloc_local((1,), "float32")
    T.ptx.match.any.sync.b32(output32[lane], source32[lane], T.uint32(0xFFFFFFFF))
    T.ptx.match.any.sync.b64(output64[lane], source64[lane], T.uint32(0xFFFFFFFF))
    T.ptx.activemask.b32(mask[0])
    active[lane] = mask[0]


@T.prim_func
def raw_ptx_movmatrix_b16(
    source: T.Buffer((32,), "uint32"),
    output: T.Buffer((32,), "uint32"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    transposed = T.alloc_local((1,), "uint32")
    T.ptx["movmatrix.sync.aligned.m8n8.trans.b16"](
        transposed[0],
        source[lane],
    )
    output[lane] = transposed[0]


def make_raw_ptx_movmatrix_case() -> tuple[dict[str, np.ndarray], np.ndarray]:
    matrix = (np.arange(64, dtype=np.uint16).reshape(8, 8) * np.uint16(251) + np.uint16(17)).astype(
        np.uint16
    )
    source_fragments = matrix.reshape(8, 4, 2)
    source = (
        source_fragments[:, :, 0].astype(np.uint32)
        | (source_fragments[:, :, 1].astype(np.uint32) << np.uint32(16))
    ).reshape(32)
    transposed_fragments = matrix.T.reshape(8, 4, 2)
    expected = (
        transposed_fragments[:, :, 0].astype(np.uint32)
        | (transposed_fragments[:, :, 1].astype(np.uint32) << np.uint32(16))
    ).reshape(32)
    return {
        "source": source,
        "output": np.zeros(32, dtype=np.uint32),
    }, expected


def test_raw_ptx_warp_collectives_match_ptx_membermask_and_bit_semantics(tmp_path):
    lanes = np.arange(32, dtype=np.uint32)
    source_bits = lanes * np.uint32(0x01010101) ^ np.uint32(0xFFC00000)
    source = source_bits.view(np.float32)
    module = numsim.transpile(raw_ptx_warp_collectives, cache_dir=tmp_path)
    result = numsim.Engine().run(
        module,
        {
            "source": source,
            "output": np.zeros((32, 6), dtype=np.uint32),
        },
    )
    output = result.outputs["output"]

    selectors = (lanes * np.uint32(7) + np.uint32(3)) % np.uint32(32)
    np.testing.assert_array_equal(output[:, 0], source_bits[selectors])
    np.testing.assert_array_equal(output[:, 1], source_bits[lanes ^ np.uint32(1)])
    expected_down = source_bits.copy()
    expected_down[:30] = source_bits[2:]
    np.testing.assert_array_equal(output[:, 2], expected_down)
    np.testing.assert_array_equal(
        output[:, 3],
        np.concatenate((np.ones(30, dtype=np.uint32), np.zeros(2, dtype=np.uint32))),
    )
    np.testing.assert_array_equal(output[:, 4], np.zeros(32, dtype=np.uint32))
    np.testing.assert_array_equal(
        output[:, 5],
        np.concatenate((np.ones(1, dtype=np.uint32), np.zeros(31, dtype=np.uint32))),
    )


def test_raw_ptx_shuffle_reads_only_selected_source_lanes(tmp_path):
    lanes = np.arange(32, dtype=np.uint32)
    initialized_sources = (lanes // np.uint32(4)) * np.uint32(4) + np.uint32(2)
    module = numsim.transpile(raw_ptx_shuffle_sparse_sources, cache_dir=tmp_path)

    selected_only = numsim.Engine().run(
        module,
        {
            "selector": initialized_sources,
            "output": np.zeros(32, dtype=np.uint32),
        },
    )
    assert selected_only.diagnostics == []
    np.testing.assert_array_equal(
        selected_only.outputs["output"],
        (initialized_sources + np.uint32(100)).astype(np.float32).view(np.uint32),
    )

    selected_uninitialized = numsim.Engine().run(
        module,
        {
            "selector": lanes,
            "output": np.zeros(32, dtype=np.uint32),
        },
    )
    assert len(selected_uninitialized.diagnostics) == 24
    assert {item["status"] for item in selected_uninitialized.diagnostics} == {"review"}
    assert {item["kind"] for item in selected_uninitialized.diagnostics} == {"uninitialized_read"}


def test_raw_ptx_f32_reductions_match_nan_and_signed_zero_semantics(tmp_path):
    source = np.arange(32, dtype=np.float32)
    source[5] = np.array([0x7FC1_2345], dtype=np.uint32).view(np.float32)[0]
    zeros = np.zeros(32, dtype=np.float32)
    zeros[1::2] = np.float32(-0.0)

    module = numsim.transpile(raw_ptx_f32_reductions, cache_dir=tmp_path)
    result = numsim.Engine().run(
        module,
        {
            "source": source,
            "zeros": zeros,
            "output": np.zeros((32, 6), dtype=np.float32),
        },
    )
    output = result.outputs["output"]

    np.testing.assert_array_equal(
        output[:, :2].view(np.uint32),
        np.full((32, 2), np.uint32(0x7FFF_FFFF), dtype=np.uint32),
    )
    np.testing.assert_array_equal(output[:, 2], np.full(32, 31.0, dtype=np.float32))
    np.testing.assert_array_equal(output[:, 3], np.zeros(32, dtype=np.float32))
    np.testing.assert_array_equal(output[:, 4].view(np.uint32), np.zeros(32, dtype=np.uint32))
    np.testing.assert_array_equal(
        output[:, 5].view(np.uint32),
        np.full(32, np.uint32(0x8000_0000), dtype=np.uint32),
    )


def test_raw_ptx_vote_and_integer_reductions_match_member_lane_oracles(tmp_path):
    lanes = np.arange(32, dtype=np.uint32)
    predicates = lanes % np.uint32(3) == 1
    values = np.uint32(0xF000_0000) + lanes * np.uint32(0x0101_0101)
    ballot = np.uint32(sum(int(predicate) << lane for lane, predicate in enumerate(predicates)))
    add = np.uint32(sum(int(value) for value in values) & 0xFFFF_FFFF)

    result = numsim.Engine().run(
        numsim.transpile(raw_ptx_vote_and_integer_reductions, cache_dir=tmp_path),
        {
            "predicates": predicates,
            "values": values,
            "output": np.zeros((32, 6), dtype=np.uint32),
        },
    )
    output = result.outputs["output"]

    np.testing.assert_array_equal(output[:, 0], np.ones(32, dtype=np.uint32))
    np.testing.assert_array_equal(output[:, 1], np.zeros(32, dtype=np.uint32))
    np.testing.assert_array_equal(output[:, 2], np.zeros(32, dtype=np.uint32))
    np.testing.assert_array_equal(output[:, 3], np.full(32, ballot, dtype=np.uint32))
    np.testing.assert_array_equal(output[:, 4], np.full(32, add, dtype=np.uint32))
    np.testing.assert_array_equal(output[:, 5], np.full(32, values.min(), dtype=np.uint32))


def test_raw_ptx_vote_accepts_tagged_uint32_predicate_carrier(tmp_path):
    predicates = np.zeros(32, dtype=np.uint32)
    predicates[[0, 5, 17, 31]] = np.array([1, 2, 0x8000_0000, 0xFFFF_FFFF], dtype=np.uint32)
    ballot = np.uint32(sum(1 << lane for lane in [0, 5, 17, 31]))

    result = numsim.Engine().run(
        numsim.transpile(raw_ptx_vote_predicate_carrier, cache_dir=tmp_path),
        {
            "predicates": predicates,
            "output_u32": np.zeros(32, dtype=np.uint32),
            "output_i32": np.zeros(32, dtype=np.int32),
            "output_f32": np.zeros(32, dtype=np.float32),
        },
    )

    for name in ("output_u32", "output_i32", "output_f32"):
        np.testing.assert_array_equal(
            result.outputs[name].view(np.uint32), np.full(32, ballot, dtype=np.uint32)
        )


def _matching_lane_masks(values: np.ndarray) -> np.ndarray:
    bits = values.view(f"uint{values.dtype.itemsize * 8}")
    return np.array(
        [
            sum(1 << source_lane for source_lane, source in enumerate(bits) if source == value)
            for value in bits
        ],
        dtype=np.uint32,
    )


def test_raw_ptx_match_redux_and_activemask_match_warp_oracles(tmp_path):
    lanes = np.arange(32, dtype=np.uint32)
    source32 = (lanes % np.uint32(5)) | (np.uint32(1) << (lanes % np.uint32(31)))
    source64 = (lanes % np.uint32(3)).astype(np.uint64) * np.uint64(0x100000001)

    module = numsim.transpile(raw_ptx_match_redux_and_activemask, cache_dir=tmp_path)
    result = numsim.Engine().run(
        module,
        {
            "source32": source32,
            "source64": source64,
            "partial_mask": np.array([0xFF], dtype=np.uint32),
            "output": np.zeros((32, 13), dtype=np.uint32),
        },
    )
    output = result.outputs["output"]

    np.testing.assert_array_equal(
        output[:, 0],
        np.concatenate((np.full(8, 0xFF, dtype=np.uint32), np.zeros(24, dtype=np.uint32))),
    )
    np.testing.assert_array_equal(output[:, 1], _matching_lane_masks(source32))
    np.testing.assert_array_equal(output[:, 2], _matching_lane_masks(source64))
    np.testing.assert_array_equal(output[:, 3], np.zeros(32, dtype=np.uint32))
    np.testing.assert_array_equal(output[:, 4:6], np.zeros((32, 2), dtype=np.uint32))
    np.testing.assert_array_equal(output[:, 6], np.full(32, 0xFFFFFFFF, dtype=np.uint32))
    np.testing.assert_array_equal(output[:, 7], np.ones(32, dtype=np.uint32))
    np.testing.assert_array_equal(
        output[:, 8], np.full(32, np.bitwise_and.reduce(source32), dtype=np.uint32)
    )
    np.testing.assert_array_equal(
        output[:, 9], np.full(32, np.bitwise_or.reduce(source32), dtype=np.uint32)
    )
    np.testing.assert_array_equal(
        output[:, 10], np.full(32, np.bitwise_xor.reduce(source32), dtype=np.uint32)
    )
    np.testing.assert_array_equal(output[:8, 11], _matching_lane_masks(source32[:8]))
    np.testing.assert_array_equal(output[8:, 11:], np.zeros((24, 2), dtype=np.uint32))
    np.testing.assert_array_equal(
        output[:8, 12], np.full(8, np.bitwise_xor.reduce(source32[:8]), dtype=np.uint32)
    )

    with pytest.raises(
        numsim.NumSimExecutionError, match="participant mask names an inactive lane"
    ):
        numsim.Engine().run(
            module,
            {
                "source32": source32,
                "source64": source64,
                "partial_mask": np.array([0xFFFFFFFF], dtype=np.uint32),
                "output": np.zeros((32, 13), dtype=np.uint32),
            },
        )
    assert {
        "tirx.ptx.activemask",
        "tirx.ptx.match_all_sync",
        "tirx.ptx.match_all_sync_p",
        "tirx.ptx.match_any_sync",
        "tirx.ptx.redux_sync_bitwise",
    } <= call_op_names(module.spec.kernels[0])


def test_raw_ptx_match_and_activemask_preserve_float_carrier_bits(tmp_path):
    source32_bits = np.resize(
        np.array([0, 0x80000000, 0x7FC12345, 0x7FC12345], dtype=np.uint32), 32
    )
    source64_bits = np.resize(
        np.array(
            [0, 0x8000000000000000, 0x7FF8000000001234, 0x7FF8000000001234],
            dtype=np.uint64,
        ),
        32,
    )
    result = numsim.Engine().run(
        numsim.transpile(raw_ptx_match_float_carriers, cache_dir=tmp_path),
        {
            "source32": source32_bits.view(np.float32),
            "source64": source64_bits.view(np.float64),
            "output32": np.zeros(32, dtype=np.int32),
            "output64": np.zeros(32, dtype=np.uint32),
            "active": np.zeros(32, dtype=np.float32),
        },
    )

    np.testing.assert_array_equal(
        result.outputs["output32"].view(np.uint32), _matching_lane_masks(source32_bits)
    )
    np.testing.assert_array_equal(result.outputs["output64"], _matching_lane_masks(source64_bits))
    np.testing.assert_array_equal(
        result.outputs["active"].view(np.uint32), np.full(32, 0xFFFFFFFF, dtype=np.uint32)
    )


@pytest.mark.parametrize(
    "checker", [_run_synccheck, _run_racecheck], ids=["synccheck", "racecheck"]
)
def test_raw_ptx_match_collectives_execute_in_checkers(tmp_path, checker):
    checker(
        raw_ptx_match_redux_and_activemask,
        {
            "source32": np.arange(32, dtype=np.uint32),
            "source64": np.arange(32, dtype=np.uint64),
            "partial_mask": np.array([0xFF], dtype=np.uint32),
            "output": np.zeros((32, 13), dtype=np.uint32),
        },
        cache_dir=tmp_path,
        max_workers=1,
    ).require_clean()


def test_raw_ptx_movmatrix_b16_matches_register_fragment_transpose(tmp_path):
    arguments, expected = make_raw_ptx_movmatrix_case()
    result = numsim.Engine().run(
        numsim.transpile(raw_ptx_movmatrix_b16, cache_dir=tmp_path),
        arguments,
    )

    np.testing.assert_array_equal(result.outputs["output"], expected)
