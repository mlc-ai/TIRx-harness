from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from tvm.script import tirx as T
from tvm.script.tirx import tile as Tx

from tests.numsim.microtests.harness import PairedTensorMap
from tests.numsim.support.kernels import (
    raw_tma_gather4_bar_address,
    raw_tma_reduce_add,
    raw_tma_roundtrip,
    tma_copy_cluster_multicast,
    tma_copy_nan_fill_boundary,
    tma_copy_roundtrip,
    tma_copy_zero_fill_boundary,
)


@T.prim_func
def typed_tma_reduce_add_float32(
    source: T.Buffer((4,), "float32"), output: T.Buffer((4,), "float32")
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((4,), "float32", scope="shared")
    if lane < 4:
        shared[lane] = source[lane]
    T.cuda.warp_sync()
    T.ptx.fence.proxy.async_.shared__cta()
    if lane == 0:
        Tx.copy_async(output[:], shared[:], dispatch="tma_auto", use_tma_reduce="add")
        T.ptx.cp.async_.bulk.commit_group()
        T.ptx.cp.async_.bulk.wait_group.read(0)


@T.prim_func
def typed_tma_reduce_min_float16(
    source: T.Buffer((8,), "float16"), output: T.Buffer((8,), "float16")
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((8,), "float16", scope="shared")
    if lane < 8:
        shared[lane] = source[lane]
    T.cuda.warp_sync()
    T.ptx.fence.proxy.async_.shared__cta()
    if lane == 0:
        Tx.copy_async(output[:], shared[:], dispatch="tma_auto", use_tma_reduce="min")
        T.ptx.cp.async_.bulk.commit_group()
        T.ptx.cp.async_.bulk.wait_group.read(0)


@T.prim_func
def typed_tma_reduce_max_float16(
    source: T.Buffer((8,), "float16"), output: T.Buffer((8,), "float16")
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((8,), "float16", scope="shared")
    if lane < 8:
        shared[lane] = source[lane]
    T.cuda.warp_sync()
    T.ptx.fence.proxy.async_.shared__cta()
    if lane == 0:
        Tx.copy_async(output[:], shared[:], dispatch="tma_auto", use_tma_reduce="max")
        T.ptx.cp.async_.bulk.commit_group()
        T.ptx.cp.async_.bulk.wait_group.read(0)


@T.prim_func
def typed_tma_reduce_add_nonzero_target(
    source: T.Buffer((4,), "float32"), output: T.Buffer((8,), "float32")
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((4,), "float32", scope="shared")
    if lane < 4:
        shared[lane] = source[lane]
    T.cuda.warp_sync()
    T.ptx.fence.proxy.async_.shared__cta()
    if lane == 0:
        start = T.meta_var(4)
        Tx.copy_async(
            output[start : start + 4],
            shared[:],
            dispatch="tma_auto",
            use_tma_reduce="add",
        )
        T.ptx.cp.async_.bulk.commit_group()
        T.ptx.cp.async_.bulk.wait_group.read(0)


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
    TmaCase("typed_roundtrip", tma_copy_roundtrip, _typed_roundtrip_arguments, ("output",)),
    TmaCase("typed_oob_zero", tma_copy_zero_fill_boundary, _boundary_arguments, ("output",)),
    TmaCase("typed_oob_nan", tma_copy_nan_fill_boundary, _boundary_arguments, ("output",)),
    TmaCase("typed_reduce_add", typed_tma_reduce_add_float32, _add_arguments, ("output",)),
    TmaCase("typed_reduce_min", typed_tma_reduce_min_float16, _minmax_arguments, ("output",)),
    TmaCase("typed_reduce_max", typed_tma_reduce_max_float16, _minmax_arguments, ("output",)),
    TmaCase(
        "typed_reduce_nonzero_target",
        typed_tma_reduce_add_nonzero_target,
        _offset_add_arguments,
        ("output",),
    ),
    TmaCase(
        "typed_multicast_cta_group2",
        tma_copy_cluster_multicast,
        _multicast_arguments,
        ("output",),
    ),
    TmaCase("raw_roundtrip", raw_tma_roundtrip, _raw_roundtrip_arguments, ("output_map",)),
    TmaCase("raw_reduce_add", raw_tma_reduce_add, _raw_reduce_arguments, ("output_map",)),
    TmaCase("raw_gather4", raw_tma_gather4_bar_address, _raw_gather4_arguments, ("output",)),
)
