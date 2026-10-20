"""Execute the baseline's per-shard GEMM and exact auxiliary protocols."""

import numpy as np
import pytest

from ported.cutlass.sm100_all_gather_gemm import cutlass_gemm, release_flag, sync_nvl_devices
from tests.numsim.support.multimem_gemm_all_reduce import _operand_bytes
from tirx_harness import numsim


@pytest.fixture(scope="module")
def kernel_cache(tmp_path_factory):
    return tmp_path_factory.mktemp("all-gather-fidelity")


@pytest.mark.parametrize("gated", (False, True))
@pytest.mark.parametrize("paired", (False, True))
def test_single_shard_matches_reference_without_a_copy_variant(gated, paired, kernel_cache):
    rows, columns, reduction = 128, 128, 128
    generator = np.random.default_rng(42)
    left = generator.integers(-1, 2, (rows, reduction))
    right = generator.integers(-1, 2, (columns, reduction))
    left_bytes = _operand_bytes(left)
    function = cutlass_gemm(
        rows, columns, reduction, "bf16", "f32", use_2cta=paired,
        mma_tiler=(256, 128) if paired else (128, 64),
        cluster=(2, 1) if paired else (1, 1), use_tma_store=True,
        max_active_clusters=4, gated_a_load=gated,
    ).func
    inputs = {
        "a": left_bytes, "scratch": left_bytes,
        "b": _operand_bytes(right), "out": np.zeros(rows * columns * 4, np.uint8),
        "flags": np.array([int(gated)], np.uint32),
    }
    result = numsim.Engine().run(numsim.transpile(function, cache_dir=kernel_cache), inputs,
                                 outputs=["out"])

    actual = result.outputs["out"].view(np.float32).reshape(rows, columns)
    np.testing.assert_array_equal(actual, left.astype(np.float64) @ right.astype(np.float64).T)


@pytest.mark.parametrize("world", (2, 4))
def test_baseline_rendezvous_uses_registered_peer_addresses(world, kernel_cache):
    counters = [np.zeros(1, np.int32) for _ in range(world)]
    counter_binding = numsim.SymmetricBuffer(counters)
    addresses = np.array([counter.ctypes.data for counter in counters], np.int64)
    inputs = [
        {"rank": np.int32(rank), "addresses": addresses.copy(),
         "flags": np.ones(world, np.int32), "counter": counter_binding}
        for rank in range(world)
    ]
    result = numsim.Engine().run(
        numsim.transpile(sync_nvl_devices(world), cache_dir=kernel_cache), inputs,
        outputs=["flags", "counter"],
    )

    for rank in range(world):
        np.testing.assert_array_equal(result.outputs[numsim.rank_binding_name("flags", rank)], 0)
        np.testing.assert_array_equal(result.outputs[numsim.rank_binding_name("counter", rank)], 0)


def test_baseline_release_updates_only_the_selected_step(kernel_cache):
    result = numsim.Engine().run(
        numsim.transpile(release_flag(4), cache_dir=kernel_cache),
        {"flags": np.zeros(4, np.int32), "step": np.int32(2)}, outputs=["flags"],
    )

    np.testing.assert_array_equal(result.outputs["flags"], [0, 0, 1, 0])
