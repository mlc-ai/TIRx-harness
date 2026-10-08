from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import ml_dtypes
import numpy as np
import tvm
from tvm.script import tirx as T

from tests.numsim.microtests.harness import PairedBuffer


@T.prim_func
def cuda_atomic_f16_global_shared(
    counter: T.Buffer((1,), "float16"),
    initial: T.Buffer((1,), "float16"),
    increment: T.Buffer((1,), "float16"),
    old_values: T.Buffer((2,), "float16"),
    final_values: T.Buffer((2,), "float16"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((1,), "float16", scope="shared")
    if lane == 0:
        shared[0] = initial[0]
    T.cuda.warp_sync()
    if lane == 0:
        old_values[0] = T.cuda.atomic_add(counter.ptr_to([0]), increment[0])
        old_values[1] = T.cuda.atomic_add(shared.ptr_to([0]), increment[0])
    T.cuda.warp_sync()
    if lane == 0:
        final_values[0] = counter[0]
        final_values[1] = shared[0]


@T.prim_func
def cuda_atomic_bf16_global_shared(
    counter: T.Buffer((1,), "bfloat16"),
    initial: T.Buffer((1,), "bfloat16"),
    increment: T.Buffer((1,), "bfloat16"),
    old_values: T.Buffer((2,), "bfloat16"),
    final_values: T.Buffer((2,), "bfloat16"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((1,), "bfloat16", scope="shared")
    if lane == 0:
        shared[0] = initial[0]
    T.cuda.warp_sync()
    if lane == 0:
        old_values[0] = T.cuda.atomic_add(counter.ptr_to([0]), increment[0])
        old_values[1] = T.cuda.atomic_add(shared.ptr_to([0]), increment[0])
    T.cuda.warp_sync()
    if lane == 0:
        final_values[0] = counter[0]
        final_values[1] = shared[0]


@T.prim_func
def cuda_atomic_f32_global_shared(
    counter: T.Buffer((1,), "float32"),
    initial: T.Buffer((1,), "float32"),
    increment: T.Buffer((1,), "float32"),
    old_values: T.Buffer((2,), "float32"),
    final_values: T.Buffer((2,), "float32"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((1,), "float32", scope="shared")
    if lane == 0:
        shared[0] = initial[0]
    T.cuda.warp_sync()
    if lane == 0:
        old_values[0] = T.cuda.atomic_add(counter.ptr_to([0]), increment[0])
        old_values[1] = T.cuda.atomic_add(shared.ptr_to([0]), increment[0])
    T.cuda.warp_sync()
    if lane == 0:
        final_values[0] = counter[0]
        final_values[1] = shared[0]


@T.prim_func
def cuda_atomic_f64_global_shared(
    counter: T.Buffer((1,), "float64"),
    initial: T.Buffer((1,), "float64"),
    increment: T.Buffer((1,), "float64"),
    old_values: T.Buffer((2,), "float64"),
    final_values: T.Buffer((2,), "float64"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((1,), "float64", scope="shared")
    if lane == 0:
        shared[0] = initial[0]
    T.cuda.warp_sync()
    if lane == 0:
        old_values[0] = T.cuda.atomic_add(counter.ptr_to([0]), increment[0])
        old_values[1] = T.cuda.atomic_add(shared.ptr_to([0]), increment[0])
    T.cuda.warp_sync()
    if lane == 0:
        final_values[0] = counter[0]
        final_values[1] = shared[0]


@T.prim_func
def cuda_atomic_f16x2_global_shared(
    counter: T.Buffer((1,), "float16x2"),
    initial: T.Buffer((1,), "float16x2"),
    increment: T.Buffer((1,), "float16x2"),
    old_values: T.Buffer((2,), "float16x2"),
    final_values: T.Buffer((2,), "float16x2"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((1,), "float16x2", scope="shared")
    if lane == 0:
        shared[0] = initial[0]
    T.cuda.warp_sync()
    if lane == 0:
        old_values[0] = T.cuda.atomic_add(counter.ptr_to([0]), increment[0])
        old_values[1] = T.cuda.atomic_add(shared.ptr_to([0]), increment[0])
    T.cuda.warp_sync()
    if lane == 0:
        final_values[0] = counter[0]
        final_values[1] = shared[0]


@T.prim_func
def cuda_atomic_bf16x2_global_shared(
    counter: T.Buffer((1,), "bfloat16x2"),
    initial: T.Buffer((1,), "bfloat16x2"),
    increment: T.Buffer((1,), "bfloat16x2"),
    old_values: T.Buffer((2,), "bfloat16x2"),
    final_values: T.Buffer((2,), "bfloat16x2"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((1,), "bfloat16x2", scope="shared")
    if lane == 0:
        shared[0] = initial[0]
    T.cuda.warp_sync()
    if lane == 0:
        old_values[0] = T.cuda.atomic_add(counter.ptr_to([0]), increment[0])
        old_values[1] = T.cuda.atomic_add(shared.ptr_to([0]), increment[0])
    T.cuda.warp_sync()
    if lane == 0:
        final_values[0] = counter[0]
        final_values[1] = shared[0]


@T.prim_func
def cuda_atomic_f32x2_global(
    counter: T.Buffer((1,), "float32x2"),
    increment: T.Buffer((1,), "float32x2"),
    old_value: T.Buffer((1,), "float32x2"),
    final_value: T.Buffer((1,), "float32x2"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    if lane == 0:
        old_value[0] = T.cuda.atomic_add(counter.ptr_to([0]), increment[0])
    T.cuda.warp_sync()
    if lane == 0:
        final_value[0] = counter[0]


@T.prim_func
def cuda_atomic_f32x4_global(
    counter: T.Buffer((1,), "float32x4"),
    increment: T.Buffer((1,), "float32x4"),
    old_value: T.Buffer((1,), "float32x4"),
    final_value: T.Buffer((1,), "float32x4"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    if lane == 0:
        old_value[0] = T.cuda.atomic_add(counter.ptr_to([0]), increment[0])
    T.cuda.warp_sync()
    if lane == 0:
        final_value[0] = counter[0]


@T.prim_func
def ptx_atomic_f32_global_shared(
    atom_cell: T.Buffer((1,), "float32"),
    red_cell: T.Buffer((1,), "float32"),
    initial: T.Buffer((1,), "float32"),
    increment: T.Buffer((1,), "float32"),
    atom_old: T.Buffer((2,), "float32"),
    atom_final: T.Buffer((2,), "float32"),
    red_final: T.Buffer((2,), "float32"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared_atom = T.alloc_buffer((1,), "float32", scope="shared")
    shared_red = T.alloc_buffer((1,), "float32", scope="shared")
    if lane == 0:
        shared_atom[0] = initial[0]
        shared_red[0] = initial[0]
    T.cuda.warp_sync()
    if lane == 0:
        T.ptx.atom.relaxed.gpu.global_.add.f32(
            atom_old[0],
            atom_cell.ptr_to([0]),
            increment[0],
        )
        T.ptx.atom.relaxed.cta.shared.add.f32(
            atom_old[1],
            shared_atom.ptr_to([0]),
            increment[0],
        )
        T.ptx.red.relaxed.gpu.global_.add.f32(
            red_cell.ptr_to([0]),
            increment[0],
        )
        T.ptx.red.relaxed.cta.shared.add.f32(
            shared_red.ptr_to([0]),
            increment[0],
        )
    T.cuda.warp_sync()
    if lane == 0:
        atom_final[0] = atom_cell[0]
        atom_final[1] = shared_atom[0]
        red_final[0] = red_cell[0]
        red_final[1] = shared_red[0]


@T.prim_func
def ptx_atomic_f64_global_shared(
    atom_cell: T.Buffer((1,), "float64"),
    red_cell: T.Buffer((1,), "float64"),
    initial: T.Buffer((1,), "float64"),
    increment: T.Buffer((1,), "float64"),
    atom_old: T.Buffer((2,), "float64"),
    atom_final: T.Buffer((2,), "float64"),
    red_final: T.Buffer((2,), "float64"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared_atom = T.alloc_buffer((1,), "float64", scope="shared")
    shared_red = T.alloc_buffer((1,), "float64", scope="shared")
    if lane == 0:
        shared_atom[0] = initial[0]
        shared_red[0] = initial[0]
    T.cuda.warp_sync()
    if lane == 0:
        T.ptx.atom.relaxed.gpu.global_.add.f64(
            atom_old[0],
            atom_cell.ptr_to([0]),
            increment[0],
        )
        T.ptx.atom.relaxed.cta.shared.add.f64(
            atom_old[1],
            shared_atom.ptr_to([0]),
            increment[0],
        )
        T.ptx.red.relaxed.gpu.global_.add.f64(
            red_cell.ptr_to([0]),
            increment[0],
        )
        T.ptx.red.relaxed.cta.shared.add.f64(
            shared_red.ptr_to([0]),
            increment[0],
        )
    T.cuda.warp_sync()
    if lane == 0:
        atom_final[0] = atom_cell[0]
        atom_final[1] = shared_atom[0]
        red_final[0] = red_cell[0]
        red_final[1] = shared_red[0]


def _bits(dtype: np.dtype[Any], *values: int) -> np.ndarray:
    unsigned = np.dtype(f"uint{np.dtype(dtype).itemsize * 8}")
    return np.asarray(values, dtype=unsigned).view(dtype)


def _bf16_buffer(values: np.ndarray) -> PairedBuffer:
    logical = np.asarray(values, dtype=ml_dtypes.bfloat16)
    return PairedBuffer(logical.view(np.uint16).copy(), "bfloat16")


def _vector_buffer(values: np.ndarray, logical_dtype: str) -> PairedBuffer:
    data_type = tvm.DataType(logical_dtype)
    lanes = int(data_type.lanes)
    base_dtype = np.dtype(logical_dtype[: -len(f"x{lanes}")])
    logical = np.asarray(values, dtype=base_dtype).reshape(-1, lanes)
    byte_width = data_type.bits * lanes // 8
    carrier_dtype = {
        4: np.dtype(np.uint32),
        8: np.dtype(np.uint64),
        16: np.dtype("V16"),
    }[byte_width]
    carrier = np.ascontiguousarray(logical).view(carrier_dtype).reshape(-1)
    return PairedBuffer(carrier, logical_dtype)


def _scalar_arguments(dtype: np.dtype[Any], initial: np.ndarray, increment: np.ndarray):
    return {
        "counter": np.asarray(initial, dtype=dtype).copy(),
        "initial": np.asarray(initial, dtype=dtype).copy(),
        "increment": np.asarray(increment, dtype=dtype).copy(),
        "old_values": np.zeros(2, dtype=dtype),
        "final_values": np.zeros(2, dtype=dtype),
    }


def _f16_arguments() -> Mapping[str, Any]:
    return _scalar_arguments(
        np.dtype(np.float16), _bits(np.dtype(np.float16), 0), _bits(np.dtype(np.float16), 1)
    )


def _bf16_arguments() -> Mapping[str, Any]:
    zero = np.asarray([0.0], dtype=ml_dtypes.bfloat16)
    subnormal = np.asarray([1], dtype=np.uint16).view(ml_dtypes.bfloat16)
    return {
        "counter": _bf16_buffer(
            zero,
        ),
        "initial": _bf16_buffer(
            zero,
        ),
        "increment": _bf16_buffer(
            subnormal,
        ),
        "old_values": _bf16_buffer(
            np.zeros(2),
        ),
        "final_values": _bf16_buffer(
            np.zeros(2),
        ),
    }


def _f32_arguments() -> Mapping[str, Any]:
    return _scalar_arguments(
        np.dtype(np.float32), _bits(np.dtype(np.float32), 0), _bits(np.dtype(np.float32), 1)
    )


def _f64_arguments() -> Mapping[str, Any]:
    return _scalar_arguments(
        np.dtype(np.float64),
        _bits(np.dtype(np.float64), 0x3FF0000000000000),
        _bits(np.dtype(np.float64), 0x3CA0000000000000),
    )


def _f16x2_arguments() -> Mapping[str, Any]:
    zero = np.zeros((1, 2), dtype=np.float16)
    increment = np.asarray([0x0001, 0x8001], dtype=np.uint16).view(np.float16).reshape(1, 2)
    return {
        "counter": _vector_buffer(
            zero,
            "float16x2",
        ),
        "initial": _vector_buffer(
            zero,
            "float16x2",
        ),
        "increment": _vector_buffer(
            increment,
            "float16x2",
        ),
        "old_values": _vector_buffer(
            np.zeros((2, 2)),
            "float16x2",
        ),
        "final_values": _vector_buffer(
            np.zeros((2, 2)),
            "float16x2",
        ),
    }


def _bf16x2_arguments() -> Mapping[str, Any]:
    zero = np.zeros((1, 2), dtype=ml_dtypes.bfloat16)
    increment = np.asarray([0x0001, 0x8001], dtype=np.uint16).view(ml_dtypes.bfloat16).reshape(1, 2)
    return {
        "counter": _vector_buffer(
            zero,
            "bfloat16x2",
        ),
        "initial": _vector_buffer(
            zero,
            "bfloat16x2",
        ),
        "increment": _vector_buffer(
            increment,
            "bfloat16x2",
        ),
        "old_values": _vector_buffer(
            np.zeros((2, 2)),
            "bfloat16x2",
        ),
        "final_values": _vector_buffer(
            np.zeros((2, 2)),
            "bfloat16x2",
        ),
    }


def _f32x2_arguments() -> Mapping[str, Any]:
    initial = np.asarray([0x00000001, 0x3F800000], dtype=np.uint32).view(np.float32)
    increment = np.asarray([0x00000000, 0x33800000], dtype=np.uint32).view(np.float32)
    return {
        "counter": _vector_buffer(
            initial,
            "float32x2",
        ),
        "increment": _vector_buffer(
            increment,
            "float32x2",
        ),
        "old_value": _vector_buffer(
            np.zeros((1, 2)),
            "float32x2",
        ),
        "final_value": _vector_buffer(
            np.zeros((1, 2)),
            "float32x2",
        ),
    }


def _f32x4_arguments() -> Mapping[str, Any]:
    initial = np.asarray([0x00000001, 0x80000001, 0x3F800000, 0xBF800000], dtype=np.uint32).view(
        np.float32
    )
    increment = np.asarray([0x00000000, 0x80000000, 0x33800000, 0xB3800000], dtype=np.uint32).view(
        np.float32
    )
    return {
        "counter": _vector_buffer(
            initial,
            "float32x4",
        ),
        "increment": _vector_buffer(
            increment,
            "float32x4",
        ),
        "old_value": _vector_buffer(
            np.zeros((1, 4)),
            "float32x4",
        ),
        "final_value": _vector_buffer(
            np.zeros((1, 4)),
            "float32x4",
        ),
    }


def _ptx_arguments(dtype: np.dtype[Any], initial: np.ndarray, increment: np.ndarray):
    return {
        "atom_cell": np.asarray(initial, dtype=dtype).copy(),
        "red_cell": np.asarray(initial, dtype=dtype).copy(),
        "initial": np.asarray(initial, dtype=dtype).copy(),
        "increment": np.asarray(increment, dtype=dtype).copy(),
        "atom_old": np.zeros(2, dtype=dtype),
        "atom_final": np.zeros(2, dtype=dtype),
        "red_final": np.zeros(2, dtype=dtype),
    }


def _ptx_f32_arguments() -> Mapping[str, Any]:
    return _ptx_arguments(
        np.dtype(np.float32), _bits(np.dtype(np.float32), 0), _bits(np.dtype(np.float32), 1)
    )


def _ptx_f64_arguments() -> Mapping[str, Any]:
    return _ptx_arguments(
        np.dtype(np.float64),
        _bits(np.dtype(np.float64), 0x3FF0000000000000),
        _bits(np.dtype(np.float64), 0x3CA0000000000000),
    )


@dataclass(frozen=True)
class FloatAtomicCase:
    name: str
    prim_func: Any
    make_arguments: Callable[[], Mapping[str, Any]]
    outputs: tuple[str, ...]


FLOAT_ATOMIC_CASES = (
    FloatAtomicCase(
        "cuda_f16_global_shared",
        cuda_atomic_f16_global_shared,
        _f16_arguments,
        ("old_values", "final_values"),
    ),
    FloatAtomicCase(
        "cuda_bf16_global_shared",
        cuda_atomic_bf16_global_shared,
        _bf16_arguments,
        ("old_values", "final_values"),
    ),
    FloatAtomicCase(
        "cuda_f32_global_shared",
        cuda_atomic_f32_global_shared,
        _f32_arguments,
        ("old_values", "final_values"),
    ),
    FloatAtomicCase(
        "cuda_f64_global_shared",
        cuda_atomic_f64_global_shared,
        _f64_arguments,
        ("old_values", "final_values"),
    ),
    FloatAtomicCase(
        "cuda_f16x2_global_shared",
        cuda_atomic_f16x2_global_shared,
        _f16x2_arguments,
        ("old_values", "final_values"),
    ),
    FloatAtomicCase(
        "cuda_bf16x2_global_shared",
        cuda_atomic_bf16x2_global_shared,
        _bf16x2_arguments,
        ("old_values", "final_values"),
    ),
    FloatAtomicCase(
        "cuda_f32x2_global",
        cuda_atomic_f32x2_global,
        _f32x2_arguments,
        ("old_value", "final_value"),
    ),
    FloatAtomicCase(
        "cuda_f32x4_global",
        cuda_atomic_f32x4_global,
        _f32x4_arguments,
        ("old_value", "final_value"),
    ),
    FloatAtomicCase(
        "ptx_f32_atom_red_global_shared",
        ptx_atomic_f32_global_shared,
        _ptx_f32_arguments,
        ("atom_old", "atom_final", "red_final"),
    ),
    FloatAtomicCase(
        "ptx_f64_atom_red_global_shared",
        ptx_atomic_f64_global_shared,
        _ptx_f64_arguments,
        ("atom_old", "atom_final", "red_final"),
    ),
)


__all__ = ["FLOAT_ATOMIC_CASES", "FloatAtomicCase"]
