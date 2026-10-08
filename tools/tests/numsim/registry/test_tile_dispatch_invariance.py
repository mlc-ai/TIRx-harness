from __future__ import annotations

import numpy as np

from tirx_harness import numsim
from tvm.script import tirx as T
from tvm.script.tirx import tile as Tx


@T.prim_func
def ordinary_dispatch_invariance(
    source: T.Buffer((32, 8), "float32"), output: T.Buffer((11, 32, 8), "float32")
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    values: T.f32[8]
    copy_default: T.f32[8]
    copy_reg: T.f32[8]
    copy_gmem_smem: T.f32[8]
    copy_fallback: T.f32[8]
    cast_default: T.f16[8]
    cast_reg: T.f16[8]
    cast_smem: T.f16[8]
    add_default: T.f32[8]
    add_reg: T.f32[8]
    add_smem: T.f32[8]
    sum_default: T.f32[1]
    sum_local: T.f32[1]
    sum_shared: T.f32[1]
    sum_packed: T.f32[1]

    for index in T.serial(8):
        values[index] = source[lane, index]
    Tx.copy(copy_default, values)
    Tx.copy(copy_reg, values, dispatch="reg")
    Tx.copy(copy_gmem_smem, values, dispatch="gmem_smem")
    Tx.copy(copy_fallback, values, dispatch="fallback")
    Tx.cast(cast_default, values)
    Tx.cast(cast_reg, values, dispatch="reg")
    Tx.cast(cast_smem, values, dispatch="smem")
    Tx.add(add_default, values, T.float32(0.25))
    Tx.add(add_reg, values, T.float32(0.25), dispatch="reg")
    Tx.add(add_smem, values, T.float32(0.25), dispatch="smem")
    Tx.sum(sum_default, values)
    Tx.sum(sum_local, values, dispatch="local")
    Tx.sum(sum_shared, values, dispatch="shared")
    Tx.sum(sum_packed, values, dispatch="packed_add_sum")

    for index in T.serial(8):
        output[0, lane, index] = copy_default[index]
        output[1, lane, index] = copy_reg[index]
        output[2, lane, index] = copy_gmem_smem[index]
        output[3, lane, index] = copy_fallback[index]
        output[4, lane, index] = T.cast(cast_default[index], "float32")
        output[5, lane, index] = T.cast(cast_reg[index], "float32")
        output[6, lane, index] = T.cast(cast_smem[index], "float32")
        output[7, lane, index] = add_default[index]
        output[8, lane, index] = add_reg[index]
        output[9, lane, index] = add_smem[index]
    output[10, lane, 0] = sum_default[0]
    output[10, lane, 1] = sum_local[0]
    output[10, lane, 2] = sum_shared[0]
    output[10, lane, 3] = sum_packed[0]


def test_ordinary_tile_dispatch_is_semantically_invariant(tmp_path):
    source = np.linspace(-2.0, 3.0, 32 * 8, dtype=np.float32).reshape(32, 8)
    output = np.zeros((11, 32, 8), dtype=np.float32)

    result = (
        numsim.Engine()
        .run(
            numsim.transpile(ordinary_dispatch_invariance, cache_dir=tmp_path),
            {"source": source, "output": output},
        )
        .outputs["output"]
    )

    def check_dispatch_invariance() -> None:
        for row in range(1, 4):
            np.testing.assert_array_equal(result[row], result[0])
        for row in range(5, 7):
            np.testing.assert_array_equal(result[row], result[4])
        for row in range(8, 10):
            np.testing.assert_array_equal(result[row], result[7])
        for column in range(1, 4):
            np.testing.assert_array_equal(result[10, :, column], result[10, :, 0])

    check_dispatch_invariance()
