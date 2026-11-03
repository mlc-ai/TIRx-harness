from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from tests.numsim.microtests.harness import PairedTensorMap
from tests.numsim.support.kernels import (
    raw_tma_gather4_bar_address,
    raw_tma_reduce_add,
    raw_tma_roundtrip,
)


def _tensor_map(
    array: np.ndarray,
    *,
    global_shape: tuple[int, ...],
    global_strides: tuple[int, ...],
    box_shape: tuple[int, ...],
) -> PairedTensorMap:
    return PairedTensorMap(
        array=array,
        global_shape=global_shape,
        global_strides=global_strides,
        box_shape=box_shape,
        element_strides=(1,) * len(global_shape),
    )


def _typed_roundtrip_arguments() -> Mapping[str, Any]:
    source = (np.arange(32, dtype=np.float16) - np.float16(7)).reshape(4, 8)
    return {"source": source, "output": np.full_like(source, np.float16(-99))}


def _boundary_arguments() -> Mapping[str, Any]:
    source = (np.arange(12, dtype=np.float32) + np.float32(100)).reshape(3, 4)
    return {"source": source, "output": np.full((4, 4), -99, dtype=np.float32)}


def _add_arguments() -> Mapping[str, Any]:
    return {
        "source": np.array([1.5, -2.0, 3.25, 4.0], dtype=np.float32),
        "output": np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32),
    }


def _minmax_arguments() -> Mapping[str, Any]:
    return {
        "source": np.array([1.5, 22.0, 3.25, 44.0, -1.0, 100.0, 7.0, 8.0], np.float16),
        "output": np.array([10.0, 20.0, 30.0, 40.0, 0.0, 99.0, 8.0, 7.0], np.float16),
    }


def _offset_add_arguments() -> Mapping[str, Any]:
    return {
        "source": np.array([0.5, 1.5, 2.5, 3.5], dtype=np.float32),
        "output": np.arange(8, dtype=np.float32) + np.float32(20),
    }


def _multicast_arguments() -> Mapping[str, Any]:
    return {
        "source": np.arange(8, dtype=np.float32) + np.float32(300),
        "output": np.full((2, 8), -1, dtype=np.float32),
    }


def _raw_roundtrip_arguments() -> Mapping[str, Any]:
    source = (np.arange(12, dtype=np.float32) + np.float32(0.25)).reshape(3, 4)
    output = np.zeros_like(source)
    metadata = {
        "global_shape": (4, 3),
        "global_strides": (16,),
        "box_shape": (4, 3),
    }
    return {
        "input_map": _tensor_map(source, **metadata),
        "output_map": _tensor_map(output, **metadata),
    }


def _raw_reduce_arguments() -> Mapping[str, Any]:
    output = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32)
    return {
        "source": np.array([1.5, -2.0, 3.25, 4.0], dtype=np.float32),
        "output_map": _tensor_map(
            output,
            global_shape=(4,),
            global_strides=(),
            box_shape=(4,),
        ),
    }


def _raw_gather4_arguments() -> Mapping[str, Any]:
    source = np.array(
        [[1000.0 + row * 10 + column for column in range(4)] for row in range(8)],
        dtype=np.float32,
    )
    return {
        "input_map": _tensor_map(
            source,
            global_shape=(4, 8),
            global_strides=(16,),
            box_shape=(4, 1),
        ),
        "output": np.zeros((4, 4), dtype=np.float32),
    }


@dataclass(frozen=True)
class TmaCase:
    name: str
    prim_func: Any
    make_arguments: Callable[[], Mapping[str, Any]]
    outputs: tuple[str, ...]


TMA_CASES = (
    TmaCase("raw_roundtrip", raw_tma_roundtrip, _raw_roundtrip_arguments, ("output_map",)),
    TmaCase("raw_reduce_add", raw_tma_reduce_add, _raw_reduce_arguments, ("output_map",)),
    TmaCase("raw_gather4", raw_tma_gather4_bar_address, _raw_gather4_arguments, ("output",)),
)
