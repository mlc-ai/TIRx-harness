from __future__ import annotations

import numpy as np
import pytest

from tirx_harness import numsim, racecheck, synccheck
from tvm_tirx.backend.cuda.lang.alloc_pool import SMEMPool
from tvm_tirx.script import tirx as T
from tvm_tirx.tirx.layout import Axis, S, TileLayout

LANES = 32
_LANE_OWNER_LAYOUT = TileLayout(S[LANES : 1 @ Axis.get("laneid")])


@T.function
def explicit_shared_strides_roundtrip(output: T.Tensor((64,), "float32")):
    T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=128))
    thread = T.cuda.thread_in_warpgroup()
    pool = SMEMPool()
    scratch = pool.alloc(
        (2, 4, 64), "float32", align=16, layout=TileLayout(S[(2, 4, 64) : (8192, 64, 1)])
    )
    pool.commit()
    if thread < 64:
        scratch[0, 0, thread] = T.cast(thread + 7, "float32")
    T.cuda.cta_sync()
    if thread < 64:
        output[thread] = scratch[0, 0, thread]


@T.function
def explicit_shared_strides_oob():
    T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=32))
    lane = T.cuda.lane_id()
    pool = SMEMPool()
    scratch = pool.alloc(
        (2, 4, 64), "float32", align=16, layout=TileLayout(S[(2, 4, 64) : (8192, 64, 1)])
    )
    pool.commit()
    if lane == 0:
        scratch[1, 0, 0] = T.float32(1)


@T.function
def strided_shared_view_beside_a_neighbor(
    index: T.Tensor((2,), "int32"), output: T.Tensor((1,), "float32")
):
    T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=32))
    lane = T.cuda.lane_id()
    pool = SMEMPool()
    # Column-major: a strided layout whose span stays inside its own allocation.
    columns = pool.alloc((4, 64), "float32", align=16, layout=TileLayout(S[(4, 64) : (1, 4)]))
    neighbor = pool.alloc((4, 64), "float32", align=16)
    pool.commit()
    if lane == 0:
        neighbor[0, 0] = T.float32(-1)
        columns[3, 63] = T.float32(5)
        output[0] = columns[index[0], index[1]]


@T.function
def nested_elect_only(output: T.Tensor((1,), "int32")):
    T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=32))
    if T.cuda.elect_sync():
        if T.cuda.elect_sync():
            output[0] = 1


@T.function
def elected_tcgen_wait_ld():
    T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=32))
    if T.cuda.elect_sync():
        T.ptx.tcgen05.wait__ld.sync.aligned()


@T.function
def elected_tcgen_wait_st():
    T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=32))
    if T.cuda.elect_sync():
        T.ptx.tcgen05.wait__st.sync.aligned()


def _bfloat16_zeros(
    shape: tuple[int, ...],
):
    return np.zeros(shape, dtype=np.uint16)


def test_explicit_shared_strides_use_executed_affine_addresses(tmp_path):
    module = numsim.transpile(explicit_shared_strides_roundtrip, cache_dir=tmp_path)
    result = numsim.Engine().run(
        module,
        {"output": np.zeros((64,), dtype=np.float32)},
    )

    np.testing.assert_array_equal(
        result.outputs["output"],
        np.arange(64, dtype=np.float32) + np.float32(7),
    )


def test_explicit_shared_strides_still_reject_an_executed_oob_address(tmp_path):
    module = numsim.transpile(explicit_shared_strides_oob, cache_dir=tmp_path)

    with pytest.raises(numsim.NumSimExecutionError, match="out of bounds|outside"):
        numsim.Engine().run(module, {})


def _neighbor_inputs(row, column):
    return {
        "index": np.array([row, column], dtype=np.int32),
        "output": np.zeros((1,), dtype=np.float32),
    }


def test_strided_shared_view_reads_inside_its_own_span(tmp_path):
    module = numsim.transpile(strided_shared_view_beside_a_neighbor, cache_dir=tmp_path)
    result = numsim.Engine().run(module, _neighbor_inputs(3, 63))

    np.testing.assert_array_equal(result.outputs["output"], np.array([5], dtype=np.float32))


def test_strided_shared_view_rejects_an_address_past_its_span_in_a_neighbor(tmp_path):
    """Only a view spanning past its owner is bounded by the owner instead of itself."""

    module = numsim.transpile(strided_shared_view_beside_a_neighbor, cache_dir=tmp_path)

    # columns[0, 64] is element 256: the first element of `neighbor`.
    with pytest.raises(numsim.NumSimExecutionError, match="out of bounds|outside"):
        numsim.Engine().run(module, _neighbor_inputs(0, 64))


@pytest.mark.parametrize(
    "checker",
    [
        pytest.param(synccheck, id="synccheck"),
        pytest.param(racecheck, id="racecheck"),
    ],
)
@pytest.mark.parametrize(
    ("kernel", "inputs"),
    [
        pytest.param(
            explicit_shared_strides_roundtrip,
            {"output": np.zeros((64,), dtype=np.float32)},
            id="explicit-shared-strides",
        ),
    ],
)
def test_reported_layouts_reach_each_semantic_checker(checker, kernel, inputs):
    checker(kernel, inputs=inputs).require_clean()


@pytest.mark.parametrize(
    "checker",
    [
        pytest.param(synccheck, id="synccheck"),
        pytest.param(racecheck, id="racecheck"),
    ],
)
@pytest.mark.parametrize(
    ("kernel", "inputs"),
    [
        pytest.param(
            nested_elect_only,
            {"output": np.zeros((1,), dtype=np.int32)},
            id="nested-elect-sync",
        ),
        pytest.param(
            elected_tcgen_wait_ld,
            {},
            id="tcgen-wait-ld-sync",
        ),
        pytest.param(
            elected_tcgen_wait_st,
            {},
            id="tcgen-wait-st-sync",
        ),
    ],
)
def test_sync_qualified_warp_operations_reject_single_lane_participation(checker, kernel, inputs):
    report = checker(kernel, inputs=inputs)

    assert report.verdict == "error"
    assert {finding.kind for finding in report.findings} == {"warp_collective_divergence"}
