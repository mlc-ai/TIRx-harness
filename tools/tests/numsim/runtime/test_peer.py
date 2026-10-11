"""Multi-rank NumSim launches over unicast peer memory: per-rank values of
peer loads, stores, atomics and bulk/TMA copies, the symmetric-memory binding
contract, and the ported all-gather GEMM."""

from __future__ import annotations

import numpy as np
import pytest

from tests.numsim.support import peer_all_gather_gemm as pag
from tests.numsim.support import peer_litmus as pl
from tests.numsim.support import peer_mega_moe as pm
from tirx_harness import numsim

WORLDS = (2, 4)
PRIVATE = "which is not symmetric memory"


@pytest.fixture(scope="module")
def peer_cache(tmp_path_factory):
    return tmp_path_factory.mktemp("numsim-peer")


@pytest.fixture(scope="module")
def compiled(peer_cache):
    cache = {}

    def get(func):
        if func not in cache:
            cache[func] = numsim.transpile(func, cache_dir=peer_cache)
        return cache[func]

    return get


def _outputs(result, name, world):
    return [result.outputs[numsim.rank_binding_name(name, rank)] for rank in range(world)]


@pytest.mark.parametrize("world", WORLDS)
def test_peer_stores_deliver_the_predecessors_vectors(world, compiled):
    result = numsim.Engine().run(compiled(pl.push), pl.push_inputs(world),
                                 outputs=["out", "inbox", "flag"])

    sources = pl.sources(world)
    for rank, (out, flag) in enumerate(zip(_outputs(result, "out", world),
                                           _outputs(result, "flag", world))):
        np.testing.assert_array_equal(out, sources[(rank - 1) % world])
        assert flag[0] == 1


@pytest.mark.parametrize("world", WORLDS)
@pytest.mark.parametrize("reuse", [0, 1])
def test_peer_loads_read_the_successors_outbox(reuse, world, compiled):
    result = numsim.Engine().run(compiled(pl.pull), pl.pull_inputs(world, reuse=reuse),
                                 outputs=["out", "outbox", "ack_flag"])

    sources = pl.sources(world)
    for rank in range(world):
        np.testing.assert_array_equal(_outputs(result, "out", world)[rank],
                                      sources[(rank + 1) % world])
        outbox = _outputs(result, "outbox", world)[rank]
        np.testing.assert_array_equal(outbox, 0 if reuse else sources[rank])
        assert _outputs(result, "ack_flag", world)[rank][0] == reuse


@pytest.mark.parametrize("world", WORLDS)
@pytest.mark.parametrize("op", ["atom_add_sys", "red_add_sys", "atom_add_gpu"])
def test_peer_additions_from_every_rank_all_land(op, world, compiled):
    result = numsim.Engine().run(compiled(pl.atomics), pl.atomics_inputs(world, op),
                                 outputs=["counter", "old"])

    total = world * (world + 1) // 2
    counters = _outputs(result, "counter", world)
    np.testing.assert_array_equal(counters[0], total)
    for rank in range(1, world):
        np.testing.assert_array_equal(counters[rank], 0)
    if op.startswith("atom"):
        # The values each lane's atomics return are the running sums of one
        # order of the ranks' additions.
        old = np.stack(_outputs(result, "old", world))
        for lane in range(pl.LANES):
            running = 0
            for rank in np.argsort(old[:, lane], kind="stable"):
                assert old[rank, lane] == running
                running += rank + 1
            assert running == total


@pytest.mark.parametrize("world", WORLDS)
def test_peer_compare_and_swap_admits_one_rank_per_word(world, compiled):
    result = numsim.Engine().run(compiled(pl.atomics), pl.atomics_inputs(world, "atom_cas_sys"),
                                 outputs=["counter", "old"])

    winners = _outputs(result, "counter", world)[0]
    old = np.stack(_outputs(result, "old", world))
    for lane, winner in enumerate(winners):
        assert 1 <= winner <= world
        assert old[winner - 1, lane] == 0
        assert (np.delete(old[:, lane], winner - 1) == winner).all()


@pytest.mark.parametrize("world", WORLDS)
@pytest.mark.parametrize("direction", [0, 1], ids=["load", "store"])
@pytest.mark.parametrize("copy", [0, 1], ids=["bulk", "tensor"])
def test_bulk_and_tma_copies_move_peer_memory(copy, direction, world, compiled):
    result = numsim.Engine().run(compiled(pl.bulk),
                                 pl.bulk_inputs(world, direction=direction, copy=copy),
                                 outputs=["out"])

    sources = pl.sources(world)
    shift = 1 if direction == 0 else -1
    for rank, out in enumerate(_outputs(result, "out", world)):
        np.testing.assert_array_equal(out, sources[(rank + shift) % world])


@pytest.mark.parametrize("kernel,inputs", [
    (pl.push, lambda: pl.push_inputs(2, symmetric=False)),
    (pl.pull, lambda: pl.pull_inputs(2, symmetric=False)),
    (pl.atomics, lambda: pl.atomics_inputs(2, "red_add_sys", symmetric=False)),
    (pl.bulk, lambda: pl.bulk_inputs(2, symmetric=False)),
    (pl.bulk, lambda: pl.bulk_inputs(2, symmetric=False, direction=1)),
], ids=["store", "load", "red", "bulk_load", "bulk_store"])
def test_a_peer_address_into_private_memory_is_a_memory_error(kernel, inputs, compiled):
    """Another GPU maps only symmetric-memory allocations: an address into a
    peer's private buffer faults on the GPU, and NumSim refuses it."""
    outputs = ["counter"] if kernel is pl.atomics else ["out"]
    with pytest.raises(numsim.NumSimExecutionError,
                       match=r"rank \d addresses rank \d's buffer '\w+@rank\d', " + PRIVATE):
        numsim.Engine().run(compiled(kernel), inputs(), outputs=outputs)


def test_a_tensor_map_over_a_peers_private_array_is_a_memory_error(compiled):
    inputs = pl.bulk_inputs(2, copy=1, tmap_symmetric=False)
    with pytest.raises(numsim.NumSimExecutionError,
                       match="TensorMap 'tmap': rank 0 addresses rank 1's buffer 'out@rank1'"):
        numsim.Engine().run(compiled(pl.bulk), inputs, outputs=["out"])


def test_atomics_on_a_ranks_own_private_buffer_are_fine(compiled):
    """Rank 0's peer address is its own replica: private memory it owns."""
    inputs = pl.atomics_inputs(1, "atom_add_sys", symmetric=False)
    result = numsim.Engine().run(compiled(pl.atomics), inputs, outputs=["counter"])
    np.testing.assert_array_equal(result.outputs[numsim.rank_binding_name("counter", 0)], 1)


def test_a_symmetric_buffer_needs_one_replica_per_rank(compiled):
    inputs = pl.atomics_inputs(2, "atom_add_sys")
    replicas = numsim.SymmetricBuffer([np.zeros(pl.LANES, np.uint32) for _ in range(3)])
    inputs[0]["counter"] = inputs[1]["counter"] = replicas
    with pytest.raises(numsim.NumSimExecutionError, match="has 3 replicas for 2 ranks"):
        numsim.Engine().run(compiled(pl.atomics), inputs, outputs=["counter"])


def test_symmetric_buffer_replicas_must_share_one_shape():
    with pytest.raises(ValueError, match="share one dtype and shape"):
        numsim.SymmetricBuffer([np.zeros(4, np.uint32), np.zeros(8, np.uint32)])


def test_symmetric_buffer_peer_offsets_are_replica_distances():
    replicas = [np.zeros(4, np.uint32) for _ in range(3)]
    offsets = numsim.SymmetricBuffer(replicas).peer_offsets(1)
    assert offsets.dtype == np.int64
    assert [replicas[1].ctypes.data + int(o) for o in offsets] == [
        r.ctypes.data for r in replicas]


ALL_GATHER_VARIANTS = {
    "tma_store_f32": {},
    "direct_store_f32": {"config": {"use_tma_store": False}},
    "tma_store_bf16": {"c": "bf16"},
    "two_cta_cluster": {"config": {"use_2cta": True, "mma_tiler": (256, 64), "cluster": (2, 1),
                                   "chunk_rows": 256, "copy_warps": 2}},
}


def all_gather_shape(world: int, variant: str) -> pag.Shape:
    overrides = ALL_GATHER_VARIANTS[variant]
    base = pag.Shape(world=world)
    return pag.Shape(world=world, c=overrides.get("c", base.c),
                     config={**base.config, **overrides.get("config", {})})


@pytest.mark.parametrize("world", WORLDS)
@pytest.mark.parametrize("variant", list(ALL_GATHER_VARIANTS))
def test_ported_all_gather_gemm_matches_the_reference(variant, world, compiled):
    shape = all_gather_shape(world, variant)
    rows, expected = pag.rank_inputs(shape)
    result = numsim.Engine().run(compiled(pag.kernel(shape)), rows, outputs=["out", "flags"])

    plan = shape.plan()
    for rank in range(world):
        out = pag.out_values(shape, _outputs(result, "out", world)[rank])
        np.testing.assert_array_equal(out, expected[rank])
        flags = _outputs(result, "flags", world)[rank].reshape(world, plan.chunks)
        np.testing.assert_array_equal(np.delete(flags, rank, axis=0), 1)
        np.testing.assert_array_equal(flags[rank], 0)


@pytest.mark.parametrize("shape", [
    pm.Shape(world=4),
    pm.Shape(world=2, num_tokens=(3, 1)),
], ids=["world4", "world2_uneven"])
def test_ported_mega_moe_matches_the_reference_on_every_rank(shape, compiled):
    """Every rank dispatches its tokens to experts on every rank through the
    peers' symmetric buffers and combines the results back."""
    rows, expected = pm.rank_inputs(shape)
    result = numsim.Engine().run(compiled(pm.kernel(shape)), rows,
                                 outputs=["y", "cumulative_local_expert_recv_stats"])

    for rank in range(shape.world):
        y = result.outputs[numsim.rank_binding_name("y", rank)]
        y = (y.astype(np.uint32) << 16).view(np.float32)
        np.testing.assert_array_equal(y, expected[rank]["y"])
        stats = result.outputs[numsim.rank_binding_name("cumulative_local_expert_recv_stats",
                                                        rank)]
        np.testing.assert_array_equal(stats, expected[rank]["cumulative_local_expert_recv_stats"])


@pytest.mark.parametrize("world", WORLDS)
def test_all_gather_barrier_clears_the_flags_and_restores_the_counter(world, compiled):
    result = numsim.Engine().run(compiled(pag.barrier(world, 8)), pag.barrier_inputs(world, 8),
                                 outputs=["flags", "counter"])

    for rank in range(world):
        np.testing.assert_array_equal(_outputs(result, "flags", world)[rank], 0)
        assert _outputs(result, "counter", world)[rank][0] == 0
