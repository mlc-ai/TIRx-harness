"""TIRx ``two_shot_all_reduce`` against CUTLASS's CuTeDSL
``all_reduce_two_shot_multimem.py`` on the same symmetric buffers.

    python -m benchmarks.multimem.two_shot_all_reduce [--world 4] [--shapes 1024x1024 ...]

The CUTLASS kernel is imported unchanged from ``$CUTLASS_DIR``; ``origins``
restates only its host wrapper, to launch on the capturing stream.
"""

from __future__ import annotations

import argparse
import json

from benchmarks.multimem import common

# The CUTLASS example's default and docstring shapes, then f32 TP activation
# all-reduces: tokens x hidden for hidden 4096 and 8192.
ORIGIN_SHAPES = [(1024, 1024), (1024, 512)]
SWEEP_SHAPES = [(m, n) for n in (4096, 8192)
                for m in (128, 256, 512, 1024, 2048, 4096, 8192, 16384)]
SHAPES = ORIGIN_SHAPES + SWEEP_SHAPES
WORKSPACES = 10
LAUNCHES = 100


def _tirx_launcher(rank: int, world: int, shape, workspace):
    import tvm

    from tests.numsim.support.multimem_kernels import TILE, two_shot_all_reduce

    func = two_shot_all_reduce(shape[0] // TILE, world, shape[1] // TILE)
    target = tvm.target.Target({"kind": "cuda", "arch": "sm_100a"})
    with target:
        executable = tvm.compile(tvm.IRModule({"main": func}), target=target,
                                 tir_pipeline="tirx")

    def launcher(ws):
        def launch():
            executable(rank, ws["in"], ws["out"], ws["in_mc"], ws["out_mc"], ws["flag"],
                       ws["flag_mc"])
        return launch

    return [launcher(ws) for ws in workspace]


def _check(name: str, launch, ws, reference) -> None:
    import torch
    import torch.distributed as dist

    ws["out"].zero_()
    torch.cuda.synchronize()
    dist.barrier()
    launch()
    torch.cuda.synchronize()
    dist.barrier()
    torch.testing.assert_close(ws["out"], reference, rtol=1e-5, atol=1e-5,
                               msg=lambda m: f"{name}: {m}")
    assert int(ws["flag"].count_nonzero()) == 0, f"{name} left its flags set"


def rank_main(rank: int, world: int, shapes: list[list[int]]) -> list[dict]:
    import torch
    import torch.distributed as dist

    from benchmarks.multimem import origins
    from tests.numsim.support.multimem_kernels import TILE

    device = torch.device("cuda", rank)
    rows = []
    for m, n in shapes:
        ctas = m * n // (TILE * TILE) // world
        generator = torch.Generator(device=device).manual_seed(1000 * rank + m + n)
        workspace = []
        for _ in range(WORKSPACES):
            ws = {}
            ws["in"], ws["in_mc"] = common.symmetric((m, n), torch.float32, device)
            ws["out"], ws["out_mc"] = common.symmetric((m, n), torch.float32, device)
            ws["flag"], ws["flag_mc"] = common.symmetric((ctas,), torch.int32, device)
            ws["in"].normal_(generator=generator)
            ws["out"].zero_()
            ws["flag"].zero_()
            workspace.append(ws)
        reference = workspace[0]["in"].clone()
        dist.all_reduce(reference)
        impls = {
            "cutlass": origins.two_shot_launchers(rank, world, workspace),
            "tirx": _tirx_launcher(rank, world, (m, n), workspace),
        }
        for name, closures in impls.items():
            _check(name, closures[0], workspace[0], reference)
        times = common.slowest_rank(common.time_launches(impls, launches=LAUNCHES, trials=9))
        rows.append({"M": m, "N": n, "MiB": m * n * 4 / 2**20,
                     "cutlass_us": times["cutlass"], "tirx_us": times["tirx"],
                     "ratio": times["tirx"] / times["cutlass"]})
        if rank == 0:
            print(json.dumps(rows[-1]), flush=True)
        del workspace, impls
        torch.cuda.synchronize()
        dist.barrier()
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--world", type=int, default=4)
    parser.add_argument("--shapes", nargs="*", default=None,
                        help="MxN shapes; default: the origin's shapes and the sweep")
    parser.add_argument("--json", default=None, help="write the rows to this file")
    args = parser.parse_args()
    shapes = ([[int(x) for x in s.split("x")] for s in args.shapes] if args.shapes
              else [list(s) for s in SHAPES])
    rows = common.spawn("benchmarks.multimem.two_shot_all_reduce:rank_main", args.world, shapes)
    print(common.table(rows, [("M", "M", "d"), ("N", "N", "d"), ("MiB", "MiB", ".1f"),
                              ("cutlass_us", "CUTLASS (us)", ".2f"),
                              ("tirx_us", "TIRx (us)", ".2f"),
                              ("ratio", "TIRx / CUTLASS", ".3f")]))
    if args.json:
        with open(args.json, "w") as f:
            json.dump(rows, f, indent=1)


if __name__ == "__main__":
    main()
