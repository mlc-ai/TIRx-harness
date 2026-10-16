"""Concrete bindings and an independent reference for the full-scale NVLS port."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ported.cutlass import sm100_gemm_all_reduce as module
from tirx_harness.numsim import MulticastWindow


@dataclass(frozen=True)
class Shape:
    world: int = 2
    rows: int = 256
    columns: int = 64
    reduction_size: int = 128
    output_type: str = "f32"
    protocol: str = "cutlass"
    config: dict = field(
        default_factory=lambda: {
            "use_2cta": False,
            "mma_tiler": (128, 64),
            "cluster": (1, 1),
            "use_tma_store": True,
            "max_active_clusters": 1,
        }
    )

    def plan(self) -> module.Plan:
        return module.plan(
            self.rows,
            self.columns,
            self.reduction_size,
            "bf16",
            self.output_type,
            self.world,
            protocol=self.protocol,
            **self.config,
        )

    @property
    def flag_len(self) -> int:
        plan = self.plan()
        return plan.m_tiles * plan.n_tiles + plan.cluster_size * plan.clusters


def kernel(shape: Shape, *, ordering: str = "sys_fenced"):
    return module.gemm_all_reduce(
        shape.rows,
        shape.columns,
        shape.reduction_size,
        "bf16",
        shape.output_type,
        None,
        shape.world,
        protocol=shape.protocol,
        ordering=ordering,
        flag_len=shape.flag_len,
        **shape.config,
    ).func


def multi_cta_shape(protocol: str, use_tma_store: bool, use_2cta: bool) -> Shape:
    return Shape(
        world=4,
        rows=1280 if use_2cta else 640,
        reduction_size=64,
        protocol=protocol,
        config={
            "use_2cta": use_2cta,
            "mma_tiler": (256, 64) if use_2cta else (128, 64),
            "cluster": (2, 1) if use_2cta else (1, 1),
            "use_tma_store": use_tma_store,
            "max_active_clusters": 2,
        },
    )


def _operand_bytes(values: np.ndarray) -> np.ndarray:
    words = values.astype(np.float32).view(np.uint32)
    return (words >> 16).astype(np.uint16).view(np.uint8).reshape(-1)


def rank_inputs(shape: Shape, *, seed: int = 0) -> tuple[list[dict], np.ndarray]:
    rng = np.random.default_rng(seed)
    left = [rng.integers(-1, 2, (shape.rows, shape.reduction_size)) for _ in range(shape.world)]
    right = [rng.integers(-1, 2, (shape.columns, shape.reduction_size)) for _ in range(shape.world)]
    expected = sum(
        left_rank.astype(np.float64) @ right_rank.astype(np.float64).T
        for left_rank, right_rank in zip(left, right)
    )
    output_bytes = shape.rows * shape.columns * module.C_BYTES[shape.output_type]
    partials = [np.zeros(output_bytes, np.uint8) for _ in range(shape.world)]
    outputs = [np.full(output_bytes, 0xFF, np.uint8) for _ in range(shape.world)]
    flags = [np.zeros(shape.flag_len, np.int32) for _ in range(shape.world)]
    partial_window = MulticastWindow(partials)
    output_window = MulticastWindow(outputs)
    flag_window = MulticastWindow(flags)
    bindings = [
        {
            "a": _operand_bytes(left[rank]),
            "b": _operand_bytes(right[rank]),
            "c": partials[rank],
            "out": outputs[rank],
            "c_mc": partial_window,
            "out_mc": output_window,
            "flag": flags[rank],
            "flag_mc": flag_window,
            "rank": np.int32(rank),
        }
        for rank in range(shape.world)
    ]
    return bindings, expected


def output_values(shape: Shape, output: np.ndarray) -> np.ndarray:
    if shape.output_type == "bf16":
        words = output.view(np.uint16).astype(np.uint32) << 16
        values = words.view(np.float32)
    else:
        values = output.view(np.float32 if shape.output_type == "f32" else np.float16)
    return values.reshape(shape.rows, shape.columns)
