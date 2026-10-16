"""NumSim-sized runs of the ported all-gather GEMM
(``ported/cutlass/sm100_all_gather_gemm.py``).

On the GPU the host schedule fills every rank's ``scratch`` with copy-engine
copies, each followed by a stream write of the chunk's flag. A NumSim launch
runs one kernel, so these runs build the port with its in-kernel copy role
(``copy_warps``), which pushes the same chunks to the same places and releases
the same flags from the SMs. Everything else -- the persistent schedule, the
TMA warp's ``.sys`` acquire and ``fence.proxy.async.global`` before it loads a
remote chunk, the MMA and the epilogue -- is the benchmarked kernel's code.

``scratch`` and ``flags`` are carved from one heap per rank and bound as
:class:`~tirx_harness.numsim.SymmetricBuffer` replicas, so one
``symm_rank_offset_<p>`` per peer reaches both, as on an NVSHMEM heap.
Operands are -1, 0 or 1, so every product and sum is exact in any order and
the output must equal the float64 reference bit for bit.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ported.cutlass import sm100_all_gather_gemm as module
from tirx_harness.numsim import SymmetricBuffer

BF16 = {-1: 0xBF80, 0: 0x0000, 1: 0x3F80}


@dataclass(frozen=True)
class Shape:
    world: int = 4
    m: int = 256
    n: int = 128
    k: int = 128
    c: str = "f32"
    config: dict = field(default_factory=lambda: {
        "use_2cta": False, "mma_tiler": (128, 64), "cluster": (1, 1), "use_tma_store": True,
        "chunk_rows": 128, "max_active_clusters": 4, "copy_warps": 1,
    })

    def plan(self) -> module.Plan:
        config = {key: value for key, value in self.config.items() if key != "copy_warps"}
        return module.plan(self.m, self.n, self.k, "bf16", self.c, self.world, **config)


def kernel(shape: Shape = Shape()):
    config = {key: tuple(value) if isinstance(value, list) else value
              for key, value in shape.config.items()}
    return module.all_gather_gemm(shape.m, shape.n, shape.k, "bf16", shape.c, None, shape.world,
                                  **config).func


def barrier(world: int, flag_len: int):
    return module.barrier_kernel(None, world, flag_len).func


def _heap(world: int, *sizes: int) -> tuple[list[np.ndarray], list[list[np.ndarray]]]:
    """Per-rank offsets and byte views carved at the same places of one heap
    per rank."""
    starts = np.cumsum([0, *[-(-size // 1024) * 1024 for size in sizes]])
    heaps = [np.zeros(int(starts[-1]) + 1024, np.uint8) for _ in range(world)]
    bases = [(-heap.ctypes.data) % 1024 for heap in heaps]
    views = [[heaps[r][bases[r] + start:][:size] for r in range(world)]
             for start, size in zip(starts, sizes)]
    addresses = [view.ctypes.data for view in views[0]]
    offsets = [np.array([address - addresses[r] for address in addresses], np.int64)
               for r in range(world)]
    return offsets, views


def _bf16(values: np.ndarray) -> np.ndarray:
    bits = np.zeros(values.shape, np.uint16)
    for value, code in BF16.items():
        bits[values == value] = code
    return bits


def warps_per_rank(shape: Shape) -> int:
    plan = shape.plan()
    return plan.clusters * plan.cluster_size * (6 + shape.config.get("copy_warps", 0))


def rank_inputs(shape: Shape = Shape(), *, seed: int = 0,
                flag_fill: int = 0) -> tuple[list[dict], list[np.ndarray]]:
    """Per-rank bindings and each rank's expected ``out`` (float32).
    ``flag_fill`` presets every flag, as a launch whose barrier did not clear
    the previous launch's flags finds them."""
    plan = shape.plan()
    world, m, n, k = shape.world, shape.m, shape.n, shape.k
    rng = np.random.default_rng(seed)
    a = [rng.integers(-1, 2, (m, k)) for _ in range(world)]
    b = [rng.integers(-1, 2, (n, k)) for _ in range(world)]
    gathered = np.concatenate(a).astype(np.float64)
    expected = [(gathered @ weight.T.astype(np.float64)).astype(np.float32) for weight in b]
    cb = module.C_BYTES[shape.c]
    offsets, (scratch, flags) = _heap(world, world * m * k * 2, plan.flags * 4)
    scratch_buffer = SymmetricBuffer(scratch)
    flag_buffer = SymmetricBuffer([view.view(np.uint32) for view in flags])
    for view in flag_buffer.replicas:
        view[:] = flag_fill
    rows = []
    for rank in range(world):
        row = {
            "a": _bf16(a[rank]).view(np.uint8).reshape(-1),
            "scratch": scratch_buffer,
            "b": _bf16(b[rank]).view(np.uint8).reshape(-1),
            "out": np.full(world * m * n * cb, 0xFF, np.uint8),
            "flags": flag_buffer,
            "rank": np.int32(rank),
        }
        if shape.config.get("copy_warps"):
            row.update({f"symm_rank_offset_{p}": offsets[rank][p] for p in range(world)})
        rows.append(row)
    return rows, expected


def out_values(shape: Shape, out: np.ndarray) -> np.ndarray:
    """A rank's ``out`` bytes as a float32 ``world * m x n`` matrix."""
    rows = shape.world * shape.m
    if shape.c == "f32":
        return out.view(np.float32).reshape(rows, shape.n)
    bits = out.view(np.uint16).astype(np.uint32) << 16
    return bits.view(np.float32).reshape(rows, shape.n)


def barrier_inputs(world: int, flag_len: int, *, offsets_of=None) -> list[dict]:
    """Per-rank bindings of ``barrier``: dirty flags to clear and a symmetric
    counter. ``offsets_of(rank, offsets)`` may rewrite a rank's peer offsets."""
    offsets, (counter, flags) = _heap(world, 4, flag_len * 4)
    counter_buffer = SymmetricBuffer([view.view(np.int32) for view in counter])
    flag_buffer = SymmetricBuffer([view.view(np.uint32) for view in flags])
    for view in flag_buffer.replicas:
        view[:] = 7
    rows = []
    for rank in range(world):
        row_offsets = offsets[rank] if offsets_of is None else offsets_of(rank, offsets[rank])
        rows.append({"flags": flag_buffer, "counter": counter_buffer, "rank": np.int32(rank),
                     **{f"symm_rank_offset_{p}": np.int64(row_offsets[p])
                        for p in range(world)}})
    return rows
