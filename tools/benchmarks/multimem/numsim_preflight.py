"""CPU preflight for the distributed two-shot NVLS benchmark kernel.

Run this before reserving GPUs:

    python -m benchmarks.multimem.numsim_preflight --world 4

The same TIRx kernel and per-rank binding shape are then used by
``two_shot_all_reduce`` for the live GPU comparison.
"""

from __future__ import annotations

import argparse

import numpy as np

from ported.cutlass.sm100_all_reduce import TILE, two_shot_all_reduce
from tirx_harness import numsim, racecheck, synccheck
from tirx_harness.numsim import MulticastWindow


def rank_inputs(
    world: int, *, tiles: int | None = None, seed: int = 0
) -> tuple[list[dict], np.ndarray]:
    """Build the complete list of NumSim bindings and its float64 reference."""

    tiles = world if tiles is None else tiles
    if world <= 0 or tiles <= 0 or tiles % world:
        raise ValueError("tiles must be a positive multiple of world")
    shape = (TILE * tiles, TILE)
    rng = np.random.default_rng(seed)
    sources = [rng.standard_normal(shape).astype(np.float32) for _ in range(world)]
    outputs = [np.zeros(shape, dtype=np.float32) for _ in range(world)]
    flags = [np.zeros(tiles // world, dtype=np.int32) for _ in range(world)]
    input_window = MulticastWindow(sources)
    output_window = MulticastWindow(outputs)
    flag_window = MulticastWindow(flags)
    bindings = [
        {
            "rank": np.int32(rank),
            "data_in": sources[rank],
            "data_out": outputs[rank],
            "in_mc": input_window,
            "out_mc": output_window,
            "flag": flags[rank],
            "flag_mc": flag_window,
        }
        for rank in range(world)
    ]
    reference = sum(
        (source.astype(np.float64) for source in sources), start=np.zeros(shape, np.float64)
    )
    return bindings, reference


def run_preflight(world: int, *, tiles: int | None = None, seed: int = 0) -> numsim.NumSimResult:
    """Run NumSim, Racecheck, and Synccheck for one distributed specialization."""

    tiles = world if tiles is None else tiles
    kernel = two_shot_all_reduce(tiles, world, 1)
    race_inputs, _ = rank_inputs(world, tiles=tiles, seed=seed)
    racecheck(kernel, race_inputs).require_clean()
    sync_inputs, _ = rank_inputs(world, tiles=tiles, seed=seed)
    synccheck(kernel, sync_inputs).require_clean()

    inputs, reference = rank_inputs(world, tiles=tiles, seed=seed)
    module = numsim.transpile(kernel)
    result = numsim.Engine().run(module, inputs, outputs=("data_out", "flag"))
    for rank in range(world):
        output = result.outputs[numsim.rank_binding_name("data_out", rank)]
        np.testing.assert_allclose(output, reference, rtol=1e-6, atol=1e-6)
        flag = result.outputs[numsim.rank_binding_name("flag", rank)]
        np.testing.assert_array_equal(flag, 0)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--world", type=int, default=4)
    parser.add_argument("--tiles", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    result = run_preflight(args.world, tiles=args.tiles, seed=args.seed)
    print(f"NumSim/Racecheck/Synccheck passed for {args.world} ranks")
    print(f"outputs: {sorted(result.outputs)}")


if __name__ == "__main__":
    main()
