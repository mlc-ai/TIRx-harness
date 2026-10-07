from __future__ import annotations

import numpy as np
import pytest

from tirx_harness import numsim
from tvm.script import tirx as T


@T.prim_func
def ptx_lop3_forms(
    a: T.Buffer((32,), "uint32"),
    b: T.Buffer((32,), "int32"),
    c: T.Buffer((32,), "float32"),
    q: T.Buffer((32,), "uint32"),
    plain: T.Buffer((32,), "float32"),
    bool_data: T.Buffer((32,), "int32"),
    bool_predicate: T.Buffer((32,), "uint32"),
    sink_predicate: T.Buffer((32,), "uint32"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    T.ptx.lop3.b32(plain[lane], a[lane], b[lane], c[lane], 0x1A)
    T.ptx.lop3.or_.b32(
        bool_data[lane],
        bool_predicate[lane],
        a[lane],
        b[lane],
        c[lane],
        0x80,
        T.ptx.pred(q[lane]),
    )
    T.ptx.lop3.and_.b32(
        sink_predicate[lane],
        a[lane],
        b[lane],
        c[lane],
        0xFE,
        T.ptx.pred(q[lane]),
    )


def _lop3_reference(a: np.ndarray, b: np.ndarray, c: np.ndarray, lut: int) -> np.ndarray:
    result = np.zeros(32, dtype=np.uint32)
    for row in range(8):
        if not (lut & (1 << row)):
            continue
        a_mask = a if row & 0b100 else np.bitwise_not(a)
        b_mask = b if row & 0b010 else np.bitwise_not(b)
        c_mask = c if row & 0b001 else np.bitwise_not(c)
        result |= a_mask & b_mask & c_mask
    return result


@pytest.mark.parametrize("checked_output", ("lop3", "lop3_bool", "lop3_bool_sink"))
def test_ptx_lop3_all_destination_shapes_match_truth_table_oracle(tmp_path, checked_output):
    lanes = np.arange(32, dtype=np.uint32)
    a = np.uint32(0xF0F0_0F0F) ^ (lanes * np.uint32(0x0101_0101))
    b_bits = np.uint32(0xCCCC_3333) ^ (lanes * np.uint32(0x0011_0101))
    c_bits = np.uint32(0xAAAA_5555) ^ (lanes * np.uint32(0x1001_0011))
    q = (lanes % np.uint32(3) == 0).astype(np.uint32)

    result = numsim.Engine().run(
        numsim.transpile(ptx_lop3_forms, cache_dir=tmp_path),
        {
            "a": a,
            "b": b_bits.view(np.int32),
            "c": c_bits.view(np.float32),
            "q": q,
            "plain": np.zeros(32, dtype=np.float32),
            "bool_data": np.zeros(32, dtype=np.int32),
            "bool_predicate": np.zeros(32, dtype=np.uint32),
            "sink_predicate": np.zeros(32, dtype=np.uint32),
        },
    )

    plain = _lop3_reference(a, b_bits, c_bits, 0x1A)
    bool_data = _lop3_reference(a, b_bits, c_bits, 0x80)
    sink_data = _lop3_reference(a, b_bits, c_bits, 0xFE)
    if checked_output == "lop3":
        np.testing.assert_array_equal(result.outputs["plain"].view(np.uint32), plain)
    elif checked_output == "lop3_bool":
        np.testing.assert_array_equal(result.outputs["bool_data"].view(np.uint32), bool_data)
        np.testing.assert_array_equal(
            result.outputs["bool_predicate"], ((bool_data != 0) | (q != 0)).astype(np.uint32)
        )
    else:
        np.testing.assert_array_equal(
            result.outputs["sink_predicate"], ((sink_data != 0) & (q != 0)).astype(np.uint32)
        )
