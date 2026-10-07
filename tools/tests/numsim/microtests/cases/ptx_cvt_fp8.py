"""TIRx kernels covering every packed PTX FP8 ``cvt`` form NumSim models.

Together the two kernels spell all twenty-four forms of the four ISA grammar
lines

    cvt.rn.satfinite{.relu}.f8x2type.f32       d, a, b;
    cvt.rn.satfinite{.relu}.f8x2type.fp16x2    d, a;
    cvt.rn{.relu}.f16x2.f8x2type               d, a;
    cvt.rn{.relu}{.satfinite}.bf16x2.f8x2type  d, a;

over ``.f8x2type = {.e4m3x2, .e5m2x2}`` and ``.fp16x2 = {.f16x2, .bf16x2}``,
each form exactly once, so one execution of each kernel can be compared against
every golden in `ptx_cvt_fp8_goldens`.

They are split by *assembler* reach, not by semantics.  The forms with a
``.bf16x2`` source or destination were introduced in PTX ISA 9.1/9.2 and need
``.target sm_100f`` or higher; TVM compiles a ``sm_100a`` CUDA target through
``nvcc -arch=sm_100a``, whose device PTX carries ``.target sm_100``, and ptxas
rejects them there.  `packed_fp8_cvt_sm100_forms` therefore holds everything
that assembles on that path and is the kernel the paired GPU microtest runs;
`packed_fp8_cvt_bf16x2_forms` holds the rest, whose ground truth comes from the
checked-in goldens (captured with an explicit ``compute_100a`` build).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
from tvm.script import tirx as T

from tests.numsim.microtests.cases.ptx_cvt_fp8_goldens import (
    BF16X2_SOURCE,
    F8X2_SOURCE,
    F16X2_SOURCE,
    F32_SOURCE,
    GOLDENS,
)

VALUE_COUNT = 256


@T.prim_func
def packed_fp8_cvt_sm100_forms(
    f32_source: T.Buffer((256,), "float32"),
    f16x2_source: T.Buffer((256,), "uint32"),
    f8x2_source: T.Buffer((256,), "uint16"),
    pack_from_f32: T.Buffer((4, 256), "uint16"),
    pack_from_f16x2: T.Buffer((4, 256), "uint16"),
    unpack_to_f16x2: T.Buffer((4, 256), "uint32"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    for step in range(8):
        index: T.let = step * 32 + lane
        following: T.let = (index + 1) % 256
        high: T.let = f32_source[index]
        low: T.let = f32_source[following]
        packed_f16x2: T.let = f16x2_source[index]
        packed_f8x2: T.let = f8x2_source[index]

        # cvt.rn.satfinite{.relu}.f8x2type.f32
        T.ptx["cvt.rn.satfinite.e4m3x2.f32"](pack_from_f32[0, index], high, low)
        T.ptx["cvt.rn.satfinite.relu.e4m3x2.f32"](pack_from_f32[1, index], high, low)
        T.ptx["cvt.rn.satfinite.e5m2x2.f32"](pack_from_f32[2, index], high, low)
        T.ptx["cvt.rn.satfinite.relu.e5m2x2.f32"](pack_from_f32[3, index], high, low)

        # cvt.rn.satfinite{.relu}.f8x2type.f16x2
        T.ptx["cvt.rn.satfinite.e4m3x2.f16x2"](pack_from_f16x2[0, index], packed_f16x2)
        T.ptx["cvt.rn.satfinite.relu.e4m3x2.f16x2"](pack_from_f16x2[1, index], packed_f16x2)
        T.ptx["cvt.rn.satfinite.e5m2x2.f16x2"](pack_from_f16x2[2, index], packed_f16x2)
        T.ptx["cvt.rn.satfinite.relu.e5m2x2.f16x2"](pack_from_f16x2[3, index], packed_f16x2)

        # cvt.rn{.relu}.f16x2.f8x2type
        T.ptx["cvt.rn.f16x2.e4m3x2"](unpack_to_f16x2[0, index], packed_f8x2)
        T.ptx["cvt.rn.relu.f16x2.e4m3x2"](unpack_to_f16x2[1, index], packed_f8x2)
        T.ptx["cvt.rn.f16x2.e5m2x2"](unpack_to_f16x2[2, index], packed_f8x2)
        T.ptx["cvt.rn.relu.f16x2.e5m2x2"](unpack_to_f16x2[3, index], packed_f8x2)


@T.prim_func
def packed_fp8_cvt_bf16x2_forms(
    bf16x2_source: T.Buffer((256,), "uint32"),
    f8x2_source: T.Buffer((256,), "uint16"),
    pack_from_bf16x2: T.Buffer((4, 256), "uint16"),
    unpack_to_bf16x2: T.Buffer((8, 256), "uint32"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    for step in range(8):
        index: T.let = step * 32 + lane
        packed_bf16x2: T.let = bf16x2_source[index]
        packed_f8x2: T.let = f8x2_source[index]

        # cvt.rn.satfinite{.relu}.f8x2type.bf16x2
        T.ptx["cvt.rn.satfinite.e4m3x2.bf16x2"](pack_from_bf16x2[0, index], packed_bf16x2)
        T.ptx["cvt.rn.satfinite.relu.e4m3x2.bf16x2"](pack_from_bf16x2[1, index], packed_bf16x2)
        T.ptx["cvt.rn.satfinite.e5m2x2.bf16x2"](pack_from_bf16x2[2, index], packed_bf16x2)
        T.ptx["cvt.rn.satfinite.relu.e5m2x2.bf16x2"](pack_from_bf16x2[3, index], packed_bf16x2)

        # cvt.rn{.relu}{.satfinite}.bf16x2.f8x2type
        T.ptx["cvt.rn.bf16x2.e4m3x2"](unpack_to_bf16x2[0, index], packed_f8x2)
        T.ptx["cvt.rn.relu.bf16x2.e4m3x2"](unpack_to_bf16x2[1, index], packed_f8x2)
        T.ptx["cvt.rn.satfinite.bf16x2.e4m3x2"](unpack_to_bf16x2[2, index], packed_f8x2)
        T.ptx["cvt.rn.relu.satfinite.bf16x2.e4m3x2"](unpack_to_bf16x2[3, index], packed_f8x2)
        T.ptx["cvt.rn.bf16x2.e5m2x2"](unpack_to_bf16x2[4, index], packed_f8x2)
        T.ptx["cvt.rn.relu.bf16x2.e5m2x2"](unpack_to_bf16x2[5, index], packed_f8x2)
        T.ptx["cvt.rn.satfinite.bf16x2.e5m2x2"](unpack_to_bf16x2[6, index], packed_f8x2)
        T.ptx["cvt.rn.relu.satfinite.bf16x2.e5m2x2"](unpack_to_bf16x2[7, index], packed_f8x2)


SM100_OUTPUTS = ("pack_from_f32", "pack_from_f16x2", "unpack_to_f16x2")
BF16X2_OUTPUTS = ("pack_from_bf16x2", "unpack_to_bf16x2")

#: golden key -> (kernel name, output buffer, row) written above.
FORM_SLOTS: Mapping[str, tuple[str, str, int]] = {
    "e4m3x2.f32": ("sm100", "pack_from_f32", 0),
    "e4m3x2.f32.relu": ("sm100", "pack_from_f32", 1),
    "e5m2x2.f32": ("sm100", "pack_from_f32", 2),
    "e5m2x2.f32.relu": ("sm100", "pack_from_f32", 3),
    "e4m3x2.f16x2": ("sm100", "pack_from_f16x2", 0),
    "e4m3x2.f16x2.relu": ("sm100", "pack_from_f16x2", 1),
    "e5m2x2.f16x2": ("sm100", "pack_from_f16x2", 2),
    "e5m2x2.f16x2.relu": ("sm100", "pack_from_f16x2", 3),
    "f16x2.e4m3x2": ("sm100", "unpack_to_f16x2", 0),
    "f16x2.e4m3x2.relu": ("sm100", "unpack_to_f16x2", 1),
    "f16x2.e5m2x2": ("sm100", "unpack_to_f16x2", 2),
    "f16x2.e5m2x2.relu": ("sm100", "unpack_to_f16x2", 3),
    "e4m3x2.bf16x2": ("bf16x2", "pack_from_bf16x2", 0),
    "e4m3x2.bf16x2.relu": ("bf16x2", "pack_from_bf16x2", 1),
    "e5m2x2.bf16x2": ("bf16x2", "pack_from_bf16x2", 2),
    "e5m2x2.bf16x2.relu": ("bf16x2", "pack_from_bf16x2", 3),
    "bf16x2.e4m3x2": ("bf16x2", "unpack_to_bf16x2", 0),
    "bf16x2.e4m3x2.relu": ("bf16x2", "unpack_to_bf16x2", 1),
    "bf16x2.e4m3x2.satfinite": ("bf16x2", "unpack_to_bf16x2", 2),
    "bf16x2.e4m3x2.relu.satfinite": ("bf16x2", "unpack_to_bf16x2", 3),
    "bf16x2.e5m2x2": ("bf16x2", "unpack_to_bf16x2", 4),
    "bf16x2.e5m2x2.relu": ("bf16x2", "unpack_to_bf16x2", 5),
    "bf16x2.e5m2x2.satfinite": ("bf16x2", "unpack_to_bf16x2", 6),
    "bf16x2.e5m2x2.relu.satfinite": ("bf16x2", "unpack_to_bf16x2", 7),
}

if set(FORM_SLOTS) != set(GOLDENS):
    raise AssertionError("kernel form slots and GPU goldens disagree")


def make_sm100_arguments() -> dict[str, Any]:
    return {
        "f32_source": F32_SOURCE.view(np.float32).copy(),
        "f16x2_source": F16X2_SOURCE.copy(),
        "f8x2_source": F8X2_SOURCE.copy(),
        "pack_from_f32": np.zeros((4, VALUE_COUNT), dtype=np.uint16),
        "pack_from_f16x2": np.zeros((4, VALUE_COUNT), dtype=np.uint16),
        "unpack_to_f16x2": np.zeros((4, VALUE_COUNT), dtype=np.uint32),
    }


def make_bf16x2_arguments() -> dict[str, Any]:
    return {
        "bf16x2_source": BF16X2_SOURCE.copy(),
        "f8x2_source": F8X2_SOURCE.copy(),
        "pack_from_bf16x2": np.zeros((4, VALUE_COUNT), dtype=np.uint16),
        "unpack_to_bf16x2": np.zeros((8, VALUE_COUNT), dtype=np.uint32),
    }


KERNELS = {
    "sm100": (packed_fp8_cvt_sm100_forms, make_sm100_arguments, SM100_OUTPUTS),
    "bf16x2": (packed_fp8_cvt_bf16x2_forms, make_bf16x2_arguments, BF16X2_OUTPUTS),
}


__all__ = [
    "BF16X2_OUTPUTS",
    "FORM_SLOTS",
    "KERNELS",
    "SM100_OUTPUTS",
    "VALUE_COUNT",
    "make_bf16x2_arguments",
    "make_sm100_arguments",
    "packed_fp8_cvt_bf16x2_forms",
    "packed_fp8_cvt_sm100_forms",
]
