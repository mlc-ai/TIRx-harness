"""Multi-rank NumSim launches over multicast windows: per-rank outputs,
``multimem`` values, and the binding contract."""

from __future__ import annotations

import numpy as np
import pytest
import tvm
from tvm.script import tirx as T

from tests.numsim.microtests.cases.multimem import CASES_BY_NAME
from tests.numsim.microtests.multigpu import run_numsim_case
from tests.numsim.support.multimem_allreduce import (
    VARIANTS,
    one_shot_all_reduce,
    rank_inputs,
    rank_sources,
)
from tirx_harness import numsim

WORLDS = (2, 4)


@pytest.fixture(scope="module")
def multimem_cache(tmp_path_factory):
    return tmp_path_factory.mktemp("numsim-multimem")


@pytest.fixture(scope="module")
def all_reduce(multimem_cache):
    return numsim.transpile(one_shot_all_reduce, cache_dir=multimem_cache)


@pytest.mark.parametrize("world", WORLDS)
def test_one_shot_all_reduce_gives_every_rank_the_sum(all_reduce, world):
    result = numsim.Engine().run(all_reduce, rank_inputs(world), outputs=["out", "flag"])

    expected = sum(rank_sources(world))
    for rank in range(world):
        np.testing.assert_array_equal(result.outputs[numsim.rank_binding_name("out", rank)], expected)
        assert result.outputs[numsim.rank_binding_name("flag", rank)][0] == world


def test_plain_access_to_a_multicast_window_is_a_memory_error(all_reduce):
    with pytest.raises(numsim.NumSimExecutionError, match="only multimem operations may access"):
        numsim.Engine().run(
            all_reduce, rank_inputs(2, **VARIANTS["plain_multicast_store"]), outputs=["out"]
        )


def test_a_window_needs_one_replica_per_rank(all_reduce):
    inputs = rank_inputs(2)
    inputs[1]["mc"] = inputs[0]["mc"] = numsim.MulticastWindow(
        [np.zeros(128, np.float32) for _ in range(3)]
    )
    with pytest.raises(numsim.NumSimExecutionError, match="has 3 replicas for 2 ranks"):
        numsim.Engine().run(all_reduce, inputs, outputs=["out"])


def test_window_replicas_must_share_one_shape():
    with pytest.raises(ValueError, match="share one dtype and shape"):
        numsim.MulticastWindow([np.zeros(4, np.float32), np.zeros(8, np.float32)])


def _case_inputs(case, world):
    return [case.rank_arguments(rank, world) for rank in range(world)]


@pytest.mark.parametrize("world", WORLDS)
@pytest.mark.parametrize("name", ["st_b64", "st_f32", "st_v4_f32"])
def test_st_writes_every_replica(name, world, multimem_cache):
    case = CASES_BY_NAME[name]
    sim = run_numsim_case(case, world, cache_dir=multimem_cache)

    stored = _case_inputs(case, world)[0]["src"]
    for rank in range(world):
        np.testing.assert_array_equal(sim[rank]["data"].view(case.lane_dtype),
                                      stored.view(case.lane_dtype))


@pytest.mark.parametrize("world", WORLDS)
def test_red_reduces_into_each_replica(world, multimem_cache):
    case = CASES_BY_NAME["red_add_u32"]
    sim = run_numsim_case(case, world, cache_dir=multimem_cache)

    inputs = _case_inputs(case, world)
    contributed = sum(i["src"].astype(np.uint64) for i in inputs)
    for rank in range(world):
        expected = ((inputs[rank]["data"].astype(np.uint64) + contributed) % 2**32).astype(np.uint32)
        np.testing.assert_array_equal(sim[rank]["data"], expected)


@pytest.mark.parametrize("world", WORLDS)
@pytest.mark.parametrize("name", ["red_add_f64", "red_add_v4_f32"])
def test_float_red_adds_exactly_and_f32_flushes_subnormals(name, world, multimem_cache):
    """The operands are small integers, exact in any order, except every
    fourth element, which holds only subnormals: f64 keeps them, and f32's
    `red.add` flushes each operand and the result to zero."""
    case = CASES_BY_NAME[name]
    sim = run_numsim_case(case, world, cache_dir=multimem_cache)

    inputs = _case_inputs(case, world)
    subnormal = np.arange(case.elements) % 4 == 3
    for rank in range(world):
        expected = inputs[rank]["data"] + sum(i["src"] for i in inputs)
        if np.dtype(case.carrier) == np.float32:
            expected[subnormal] = 0.0
        np.testing.assert_array_equal(np.abs(sim[rank]["data"]), np.abs(expected))


@pytest.mark.parametrize("world", WORLDS)
def test_ld_reduce_returns_the_same_sum_on_every_rank(world, multimem_cache):
    case = CASES_BY_NAME["ldr_min_s64"]
    sim = run_numsim_case(case, world, cache_dir=multimem_cache)

    expected = np.minimum.reduce([i["data"] for i in _case_inputs(case, world)])
    for rank in range(world):
        np.testing.assert_array_equal(sim[rank]["out"], expected)


@T.prim_func
def ld_reduce_f32(mc: T.Buffer((4,), "float32"), out: T.Buffer((4,), "float32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    if lane == 0:
        T.ptx.multimem_ld_reduce.global_.add.v4.f32(out[0], out[1], out[2], out[3], mc.ptr_to([0]))


@T.prim_func
def rank_queries(out: T.Buffer((2,), "int32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    if lane == 0:
        out[0] = T.nvshmem.my_pe()
        out[1] = T.nvshmem.n_pes()


@pytest.mark.parametrize("world", (1, *WORLDS))
def test_nvshmem_rank_queries_report_the_launch(world, multimem_cache):
    module = numsim.transpile(rank_queries, cache_dir=multimem_cache)
    inputs = [{"out": np.zeros(2, np.int32)} for _ in range(world)]

    result = numsim.Engine().run(module, inputs if world > 1 else inputs[0], outputs=["out"])

    if world == 1:
        np.testing.assert_array_equal(result.outputs["out"], [0, 1])
    for rank in range(world if world > 1 else 0):
        np.testing.assert_array_equal(
            result.outputs[numsim.rank_binding_name("out", rank)], [rank, world]
        )


@T.prim_func
def ld_reduce_f32_scalar(mc: T.Buffer((4,), "float32"), out: T.Buffer((4,), "float32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    if lane < 4:
        T.ptx.multimem_ld_reduce.weak.global_.add.f32(out[lane], mc.ptr_to([lane]))


@pytest.mark.parametrize("kernel", [ld_reduce_f32, ld_reduce_f32_scalar], ids=["v4", "scalar"])
def test_float_ld_reduce_models_the_nvls_accumulator(kernel, multimem_cache):
    p = lambda exponent: np.float32(2.0**exponent)  # noqa: E731
    # Columns are elements; rows are ranks.
    replicas = np.array(
        [
            [p(24), p(17), p(0), -0.0],
            [p(0), -p(17), -p(0), -0.0],
            [-p(24), p(-70), np.float32(1.75) * p(-87), -0.0],
            [p(0), 0.0, 0.0, -0.0],
        ],
        np.float32,
    )
    window = numsim.MulticastWindow([row.copy() for row in replicas])
    module = numsim.transpile(kernel, cache_dir=multimem_cache)
    result = numsim.Engine().run(
        module,
        [{"mc": window, "out": np.zeros(4, np.float32)} for _ in range(4)],
        outputs=["out"],
    )

    out = result.outputs[numsim.rank_binding_name("out", 0)]
    # The exact sum, not a rank-order fold (which loses both 1s); 2^-70 is
    # below the window anchored at 2^32; a sum below 2^-86 under anchor 0
    # reads as zero, and the accumulator has no -0.
    np.testing.assert_array_equal(out.view(np.uint32), np.array([2.0, 0.0, 0.0, 0.0], np.float32).view(np.uint32))
