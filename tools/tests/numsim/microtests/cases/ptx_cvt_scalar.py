"""Kernels covering the scalar PTX ``cvt`` grammar forms NumSim models.

Each buffer column is one exact PTX spelling; `SCALAR_CVT_COLUMNS` names it so
the recorded GPU goldens in `ptx_cvt_scalar_goldens.py` can be read back per
instruction rather than per array offset.
"""

from __future__ import annotations

import numpy as np
from tvm.script import tirx as T

from .ptx_cvt_scalar_goldens import SCALAR_CVT_INPUTS

_LANES = 16


@T.prim_func
def scalar_cvt_float_to_int(
    source_f32: T.Buffer((16,), "float32"),
    source_f64: T.Buffer((16,), "float64"),
    source_f16: T.Buffer((16,), "uint16"),
    source_bf16: T.Buffer((16,), "uint16"),
    out_s32: T.Buffer((16, 10), "int32"),
    out_u32: T.Buffer((16, 2), "uint32"),
    out_s64: T.Buffer((16, 2), "int64"),
    out_u8: T.Buffer((16, 2), "uint8"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    if lane < 16:
        T.ptx["cvt.rni.s32.f32"](out_s32[lane, 0], source_f32[lane])
        T.ptx["cvt.rzi.s32.f32"](out_s32[lane, 1], source_f32[lane])
        T.ptx["cvt.rmi.s32.f32"](out_s32[lane, 2], source_f32[lane])
        T.ptx["cvt.rpi.s32.f32"](out_s32[lane, 3], source_f32[lane])
        T.ptx["cvt.rmi.ftz.s32.f32"](out_s32[lane, 4], source_f32[lane])
        T.ptx["cvt.rpi.ftz.s32.f32"](out_s32[lane, 5], source_f32[lane])
        T.ptx["cvt.rzi.sat.s32.f32"](out_s32[lane, 6], source_f32[lane])
        T.ptx["cvt.rzi.s32.f64"](out_s32[lane, 7], source_f64[lane])
        T.ptx["cvt.rzi.s32.f16"](out_s32[lane, 8], source_f16[lane])
        T.ptx["cvt.rzi.s32.bf16"](out_s32[lane, 9], source_bf16[lane])
        T.ptx["cvt.rzi.u32.f32"](out_u32[lane, 0], source_f32[lane])
        T.ptx["cvt.rpi.u32.f64"](out_u32[lane, 1], source_f64[lane])
        T.ptx["cvt.rzi.s64.f32"](out_s64[lane, 0], source_f32[lane])
        T.ptx["cvt.rni.s64.f64"](out_s64[lane, 1], source_f64[lane])
        T.ptx["cvt.rzi.u8.f32"](out_u8[lane, 0], source_f32[lane])
        T.ptx["cvt.rzi.u8.f64"](out_u8[lane, 1], source_f64[lane])


@T.prim_func
def scalar_cvt_rounded_conversions(
    source_s64: T.Buffer((16,), "int64"),
    source_u64: T.Buffer((16,), "uint64"),
    source_f64: T.Buffer((16,), "float64"),
    source_f32: T.Buffer((16,), "float32"),
    source_bf16: T.Buffer((16,), "uint16"),
    out_f32: T.Buffer((16, 16), "float32"),
    out_f32_integral: T.Buffer((16, 6), "float32"),
    out_f64: T.Buffer((16, 6), "float64"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    if lane < 16:
        T.ptx["cvt.rn.f32.s64"](out_f32[lane, 0], source_s64[lane])
        T.ptx["cvt.rz.f32.s64"](out_f32[lane, 1], source_s64[lane])
        T.ptx["cvt.rm.f32.s64"](out_f32[lane, 2], source_s64[lane])
        T.ptx["cvt.rp.f32.s64"](out_f32[lane, 3], source_s64[lane])
        T.ptx["cvt.rn.f32.u64"](out_f32[lane, 4], source_u64[lane])
        T.ptx["cvt.rz.f32.u64"](out_f32[lane, 5], source_u64[lane])
        T.ptx["cvt.rm.f32.u64"](out_f32[lane, 6], source_u64[lane])
        T.ptx["cvt.rp.f32.u64"](out_f32[lane, 7], source_u64[lane])
        T.ptx["cvt.rn.f32.f64"](out_f32[lane, 8], source_f64[lane])
        T.ptx["cvt.rz.f32.f64"](out_f32[lane, 9], source_f64[lane])
        T.ptx["cvt.rm.f32.f64"](out_f32[lane, 10], source_f64[lane])
        T.ptx["cvt.rp.f32.f64"](out_f32[lane, 11], source_f64[lane])
        T.ptx["cvt.rp.ftz.f32.f64"](out_f32[lane, 12], source_f64[lane])
        T.ptx["cvt.ftz.f32.bf16"](out_f32[lane, 13], source_bf16[lane])
        # `.ftz` on an integer source is inert and collapses onto the plain
        # twin's specialization; these must still reproduce hardware.
        T.ptx["cvt.rz.ftz.f32.s64"](out_f32[lane, 14], source_s64[lane])
        T.ptx["cvt.rp.ftz.f32.u64"](out_f32[lane, 15], source_u64[lane])
        T.ptx["cvt.rni.f32.f32"](out_f32_integral[lane, 0], source_f32[lane])
        T.ptx["cvt.rzi.f32.f32"](out_f32_integral[lane, 1], source_f32[lane])
        T.ptx["cvt.rmi.f32.f32"](out_f32_integral[lane, 2], source_f32[lane])
        T.ptx["cvt.rpi.f32.f32"](out_f32_integral[lane, 3], source_f32[lane])
        T.ptx["cvt.rpi.ftz.f32.f32"](out_f32_integral[lane, 4], source_f32[lane])
        T.ptx["cvt.ftz.f32.f32"](out_f32_integral[lane, 5], source_f32[lane])
        T.ptx["cvt.rn.f64.u64"](out_f64[lane, 0], source_u64[lane])
        T.ptx["cvt.rz.f64.u64"](out_f64[lane, 1], source_u64[lane])
        T.ptx["cvt.rm.f64.u64"](out_f64[lane, 2], source_u64[lane])
        T.ptx["cvt.rp.f64.u64"](out_f64[lane, 3], source_u64[lane])
        T.ptx["cvt.rni.f64.f64"](out_f64[lane, 4], source_f64[lane])
        T.ptx["cvt.ftz.f64.f32"](out_f64[lane, 5], source_f32[lane])


@T.prim_func
def scalar_cvt_narrowing(
    source_f32: T.Buffer((16,), "float32"),
    out_f16: T.Buffer((16, 8), "uint16"),
    out_bf16: T.Buffer((16, 8), "uint16"),
    out_tf32: T.Buffer((16, 10), "uint32"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    if lane < 16:
        T.ptx["cvt.rn.f16.f32"](out_f16[lane, 0], source_f32[lane])
        T.ptx["cvt.rn.relu.f16.f32"](out_f16[lane, 1], source_f32[lane])
        T.ptx["cvt.rn.satfinite.f16.f32"](out_f16[lane, 2], source_f32[lane])
        T.ptx["cvt.rn.relu.satfinite.f16.f32"](out_f16[lane, 3], source_f32[lane])
        T.ptx["cvt.rz.f16.f32"](out_f16[lane, 4], source_f32[lane])
        T.ptx["cvt.rz.relu.f16.f32"](out_f16[lane, 5], source_f32[lane])
        T.ptx["cvt.rz.satfinite.f16.f32"](out_f16[lane, 6], source_f32[lane])
        T.ptx["cvt.rz.relu.satfinite.f16.f32"](out_f16[lane, 7], source_f32[lane])
        T.ptx["cvt.rn.bf16.f32"](out_bf16[lane, 0], source_f32[lane])
        T.ptx["cvt.rn.relu.bf16.f32"](out_bf16[lane, 1], source_f32[lane])
        T.ptx["cvt.rn.satfinite.bf16.f32"](out_bf16[lane, 2], source_f32[lane])
        T.ptx["cvt.rn.relu.satfinite.bf16.f32"](out_bf16[lane, 3], source_f32[lane])
        T.ptx["cvt.rz.bf16.f32"](out_bf16[lane, 4], source_f32[lane])
        T.ptx["cvt.rz.relu.bf16.f32"](out_bf16[lane, 5], source_f32[lane])
        T.ptx["cvt.rz.satfinite.bf16.f32"](out_bf16[lane, 6], source_f32[lane])
        T.ptx["cvt.rz.relu.satfinite.bf16.f32"](out_bf16[lane, 7], source_f32[lane])
        T.ptx["cvt.rn.tf32.f32"](out_tf32[lane, 0], source_f32[lane])
        T.ptx["cvt.rn.satfinite.tf32.f32"](out_tf32[lane, 1], source_f32[lane])
        T.ptx["cvt.rn.relu.tf32.f32"](out_tf32[lane, 2], source_f32[lane])
        T.ptx["cvt.rn.satfinite.relu.tf32.f32"](out_tf32[lane, 3], source_f32[lane])
        T.ptx["cvt.rz.tf32.f32"](out_tf32[lane, 4], source_f32[lane])
        T.ptx["cvt.rz.satfinite.tf32.f32"](out_tf32[lane, 5], source_f32[lane])
        T.ptx["cvt.rz.relu.tf32.f32"](out_tf32[lane, 6], source_f32[lane])
        T.ptx["cvt.rz.satfinite.relu.tf32.f32"](out_tf32[lane, 7], source_f32[lane])
        T.ptx["cvt.rna.tf32.f32"](out_tf32[lane, 8], source_f32[lane])
        T.ptx["cvt.rna.satfinite.tf32.f32"](out_tf32[lane, 9], source_f32[lane])


# One exact PTX spelling per (kernel, buffer, column).
SCALAR_CVT_COLUMNS: dict[str, dict[str, tuple[str, ...]]] = {
    "scalar_cvt_float_to_int": {
        "out_s32": (
            "cvt.rni.s32.f32",
            "cvt.rzi.s32.f32",
            "cvt.rmi.s32.f32",
            "cvt.rpi.s32.f32",
            "cvt.rmi.ftz.s32.f32",
            "cvt.rpi.ftz.s32.f32",
            "cvt.rzi.sat.s32.f32",
            "cvt.rzi.s32.f64",
            "cvt.rzi.s32.f16",
            "cvt.rzi.s32.bf16",
        ),
        "out_u32": ("cvt.rzi.u32.f32", "cvt.rpi.u32.f64"),
        "out_s64": ("cvt.rzi.s64.f32", "cvt.rni.s64.f64"),
        "out_u8": ("cvt.rzi.u8.f32", "cvt.rzi.u8.f64"),
    },
    "scalar_cvt_rounded_conversions": {
        "out_f32": (
            "cvt.rn.f32.s64",
            "cvt.rz.f32.s64",
            "cvt.rm.f32.s64",
            "cvt.rp.f32.s64",
            "cvt.rn.f32.u64",
            "cvt.rz.f32.u64",
            "cvt.rm.f32.u64",
            "cvt.rp.f32.u64",
            "cvt.rn.f32.f64",
            "cvt.rz.f32.f64",
            "cvt.rm.f32.f64",
            "cvt.rp.f32.f64",
            "cvt.rp.ftz.f32.f64",
            "cvt.ftz.f32.bf16",
            "cvt.rz.ftz.f32.s64",
            "cvt.rp.ftz.f32.u64",
        ),
        "out_f32_integral": (
            "cvt.rni.f32.f32",
            "cvt.rzi.f32.f32",
            "cvt.rmi.f32.f32",
            "cvt.rpi.f32.f32",
            "cvt.rpi.ftz.f32.f32",
            "cvt.ftz.f32.f32",
        ),
        "out_f64": (
            "cvt.rn.f64.u64",
            "cvt.rz.f64.u64",
            "cvt.rm.f64.u64",
            "cvt.rp.f64.u64",
            "cvt.rni.f64.f64",
            "cvt.ftz.f64.f32",
        ),
    },
    "scalar_cvt_narrowing": {
        "out_f16": (
            "cvt.rn.f16.f32",
            "cvt.rn.relu.f16.f32",
            "cvt.rn.satfinite.f16.f32",
            "cvt.rn.relu.satfinite.f16.f32",
            "cvt.rz.f16.f32",
            "cvt.rz.relu.f16.f32",
            "cvt.rz.satfinite.f16.f32",
            "cvt.rz.relu.satfinite.f16.f32",
        ),
        "out_bf16": (
            "cvt.rn.bf16.f32",
            "cvt.rn.relu.bf16.f32",
            "cvt.rn.satfinite.bf16.f32",
            "cvt.rn.relu.satfinite.bf16.f32",
            "cvt.rz.bf16.f32",
            "cvt.rz.relu.bf16.f32",
            "cvt.rz.satfinite.bf16.f32",
            "cvt.rz.relu.satfinite.bf16.f32",
        ),
        "out_tf32": (
            "cvt.rn.tf32.f32",
            "cvt.rn.satfinite.tf32.f32",
            "cvt.rn.relu.tf32.f32",
            "cvt.rn.satfinite.relu.tf32.f32",
            "cvt.rz.tf32.f32",
            "cvt.rz.satfinite.tf32.f32",
            "cvt.rz.relu.tf32.f32",
            "cvt.rz.satfinite.relu.tf32.f32",
            "cvt.rna.tf32.f32",
            "cvt.rna.satfinite.tf32.f32",
        ),
    },
}

SCALAR_CVT_KERNELS = {
    "scalar_cvt_float_to_int": scalar_cvt_float_to_int,
    "scalar_cvt_rounded_conversions": scalar_cvt_rounded_conversions,
    "scalar_cvt_narrowing": scalar_cvt_narrowing,
}

_OUTPUT_DTYPES = {
    "out_s32": np.int32,
    "out_u32": np.uint32,
    "out_s64": np.int64,
    "out_u8": np.uint8,
    "out_f32": np.float32,
    "out_f32_integral": np.float32,
    "out_f64": np.float64,
    "out_f16": np.uint16,
    "out_bf16": np.uint16,
    "out_tf32": np.uint32,
}

_SOURCE_DTYPES = {
    "source_f32": ("f32", np.uint32, np.float32),
    "source_f64": ("f64", np.uint64, np.float64),
    "source_f16": ("f16", np.uint16, np.uint16),
    "source_bf16": ("bf16", np.uint16, np.uint16),
    "source_s64": ("s64", np.uint64, np.int64),
    "source_u64": ("u64", np.uint64, np.uint64),
}


def scalar_cvt_arguments(name: str) -> dict[str, np.ndarray]:
    """Build the host state one scalar-cvt kernel runs from."""

    columns = SCALAR_CVT_COLUMNS[name]
    arguments: dict[str, np.ndarray] = {}
    for parameter, (source, raw, logical) in _SOURCE_DTYPES.items():
        if parameter not in _kernel_parameters(name):
            continue
        arguments[parameter] = np.asarray(SCALAR_CVT_INPUTS[source], dtype=raw).view(logical)
    for buffer, spellings in columns.items():
        arguments[buffer] = np.zeros((_LANES, len(spellings)), _OUTPUT_DTYPES[buffer])
    return arguments


def _kernel_parameters(name: str) -> tuple[str, ...]:
    return tuple(str(parameter.name) for parameter in SCALAR_CVT_KERNELS[name].params)


__all__ = [
    "SCALAR_CVT_COLUMNS",
    "SCALAR_CVT_KERNELS",
    "scalar_cvt_arguments",
    "scalar_cvt_float_to_int",
    "scalar_cvt_narrowing",
    "scalar_cvt_rounded_conversions",
]
