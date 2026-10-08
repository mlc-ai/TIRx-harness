"""TIRx kernels covering the FP4, ``.rs``, ``ue8m0``, and ``.scaled`` PTX ``cvt`` forms.

Together the two kernels spell all thirty-nine forms this wave models, each
exactly once, so one execution of each can be compared against every golden in
`ptx_cvt_narrow_goldens`:

    cvt.rn.satfinite{.relu}.e2m1x2.{f32,f16x2,bf16x2}             (6)
    cvt.rn{.relu}.f16x2.e2m1x2                                    (2)
    cvt.rn{.relu}{.satfinite}.bf16x2.e2m1x2                       (4)
    cvt.rs{.relu}.satfinite.{e2m1x4,e4m3x4,e5m2x4}.f32            (6)
    cvt.{rz,rp}{.satfinite}.ue8m0x2.{f32,bf16x2}                  (8)
    cvt.rn.bf16x2.ue8m0x2                                         (1)
    cvt.rn{.relu}{.satfinite}.scaled::n2::ue8m0.bf16x2.<x2type>  (12)

They are split by *assembler* reach, not by semantics.  `narrow_cvt_sm100_forms`
holds the twenty-one forms TVM's ``sm_100a`` compile path can assemble, and is
the kernel the paired GPU microtest runs.  `narrow_cvt_bf16x2_forms` holds the
eighteen it cannot: every form with a ``.bf16x2`` *destination* fed by a narrow
type, and every ``.scaled::n2::ue8m0`` spelling, which that path rejects with
"Unexpected instruction types specified for 'cvt'" and "Unknown modifier
'.scaled::n2::ue8m0'" respectively.  Their ground truth comes from the checked-in
goldens, captured with an explicit ``compute_100a`` build.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
from tvm.script import tirx as T

from tests.numsim.microtests.cases.ptx_cvt_narrow_goldens import (
    BF16X2_SOURCE,
    F4X2_SOURCE,
    F8X2_SOURCE,
    F16X2_SOURCE,
    F32_SOURCE,
    GOLDENS,
    RBITS_SOURCE,
    SCALE_SOURCE,
    UE8X2_SOURCE,
)

VALUE_COUNT = 256


@T.prim_func
def narrow_cvt_sm100_forms(
    f32_source: T.Buffer((256,), "float32"),
    f16x2_source: T.Buffer((256,), "uint32"),
    bf16x2_source: T.Buffer((256,), "uint32"),
    f4x2_source: T.Buffer((256,), "uint16"),
    ue8x2_source: T.Buffer((256,), "uint16"),
    rbits_source: T.Buffer((256,), "uint32"),
    pack_f4: T.Buffer((4, 256), "uint8"),
    unpack_f4: T.Buffer((2, 256), "uint32"),
    stochastic_f8: T.Buffer((4, 256), "uint32"),
    stochastic_f4: T.Buffer((2, 256), "uint16"),
    exponent: T.Buffer((8, 256), "uint16"),
    from_exponent: T.Buffer((1, 256), "uint32"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    for step in range(8):
        index: T.let = step * 32 + lane
        second: T.let = (index + 1) % 256
        third: T.let = (index + 2) % 256
        fourth: T.let = (index + 3) % 256
        a: T.let = f32_source[index]
        b: T.let = f32_source[second]
        e: T.let = f32_source[third]
        f: T.let = f32_source[fourth]
        rbits: T.let = rbits_source[index]
        packed_f4: T.let = T.cast(f4x2_source[index], "uint8")

        # cvt.rn.satfinite{.relu}.e2m1x2.f32
        T.ptx["cvt.rn.satfinite.e2m1x2.f32"](pack_f4[0, index], a, b)
        T.ptx["cvt.rn.satfinite.relu.e2m1x2.f32"](pack_f4[1, index], a, b)

        # cvt.rn.satfinite{.relu}.e2m1x2.f16x2
        T.ptx["cvt.rn.satfinite.e2m1x2.f16x2"](pack_f4[2, index], f16x2_source[index])
        T.ptx["cvt.rn.satfinite.relu.e2m1x2.f16x2"](pack_f4[3, index], f16x2_source[index])

        # cvt.rn{.relu}.f16x2.e2m1x2
        T.ptx["cvt.rn.f16x2.e2m1x2"](unpack_f4[0, index], packed_f4)
        T.ptx["cvt.rn.relu.f16x2.e2m1x2"](unpack_f4[1, index], packed_f4)

        # cvt.rs{.relu}.satfinite.{e4m3x4,e5m2x4}.f32
        T.ptx["cvt.rs.satfinite.e4m3x4.f32"](stochastic_f8[0, index], a, b, e, f, rbits)
        T.ptx["cvt.rs.relu.satfinite.e4m3x4.f32"](stochastic_f8[1, index], a, b, e, f, rbits)
        T.ptx["cvt.rs.satfinite.e5m2x4.f32"](stochastic_f8[2, index], a, b, e, f, rbits)
        T.ptx["cvt.rs.relu.satfinite.e5m2x4.f32"](stochastic_f8[3, index], a, b, e, f, rbits)

        # cvt.rs{.relu}.satfinite.e2m1x4.f32
        T.ptx["cvt.rs.satfinite.e2m1x4.f32"](stochastic_f4[0, index], a, b, e, f, rbits)
        T.ptx["cvt.rs.relu.satfinite.e2m1x4.f32"](stochastic_f4[1, index], a, b, e, f, rbits)

        # cvt.{rz,rp}{.satfinite}.ue8m0x2.f32
        T.ptx["cvt.rz.ue8m0x2.f32"](exponent[0, index], a, b)
        T.ptx["cvt.rz.satfinite.ue8m0x2.f32"](exponent[1, index], a, b)
        T.ptx["cvt.rp.ue8m0x2.f32"](exponent[2, index], a, b)
        T.ptx["cvt.rp.satfinite.ue8m0x2.f32"](exponent[3, index], a, b)

        # cvt.{rz,rp}{.satfinite}.ue8m0x2.bf16x2
        T.ptx["cvt.rz.ue8m0x2.bf16x2"](exponent[4, index], bf16x2_source[index])
        T.ptx["cvt.rz.satfinite.ue8m0x2.bf16x2"](exponent[5, index], bf16x2_source[index])
        T.ptx["cvt.rp.ue8m0x2.bf16x2"](exponent[6, index], bf16x2_source[index])
        T.ptx["cvt.rp.satfinite.ue8m0x2.bf16x2"](exponent[7, index], bf16x2_source[index])

        # cvt.rn.bf16x2.ue8m0x2
        T.ptx["cvt.rn.bf16x2.ue8m0x2"](from_exponent[0, index], ue8x2_source[index])


@T.prim_func
def narrow_cvt_bf16x2_forms(
    bf16x2_source: T.Buffer((256,), "uint32"),
    f4x2_source: T.Buffer((256,), "uint16"),
    f8x2_source: T.Buffer((256,), "uint16"),
    scale_source: T.Buffer((256,), "uint16"),
    pack_f4: T.Buffer((2, 256), "uint8"),
    unpack_f4: T.Buffer((4, 256), "uint32"),
    scaled: T.Buffer((12, 256), "uint32"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    for step in range(8):
        index: T.let = step * 32 + lane
        packed_f4: T.let = T.cast(f4x2_source[index], "uint8")
        packed_f8: T.let = f8x2_source[index]
        scale: T.let = scale_source[index]

        # cvt.rn.satfinite{.relu}.e2m1x2.bf16x2
        T.ptx["cvt.rn.satfinite.e2m1x2.bf16x2"](pack_f4[0, index], bf16x2_source[index])
        T.ptx["cvt.rn.satfinite.relu.e2m1x2.bf16x2"](pack_f4[1, index], bf16x2_source[index])

        # cvt.rn{.relu}{.satfinite}.bf16x2.e2m1x2
        T.ptx["cvt.rn.bf16x2.e2m1x2"](unpack_f4[0, index], packed_f4)
        T.ptx["cvt.rn.relu.bf16x2.e2m1x2"](unpack_f4[1, index], packed_f4)
        T.ptx["cvt.rn.satfinite.bf16x2.e2m1x2"](unpack_f4[2, index], packed_f4)
        T.ptx["cvt.rn.relu.satfinite.bf16x2.e2m1x2"](unpack_f4[3, index], packed_f4)

        # cvt.rn{.relu}{.satfinite}.scaled::n2::ue8m0.bf16x2.e4m3x2
        T.ptx["cvt.rn.scaled::n2::ue8m0.bf16x2.e4m3x2"](scaled[0, index], packed_f8, scale)
        T.ptx["cvt.rn.relu.scaled::n2::ue8m0.bf16x2.e4m3x2"](scaled[1, index], packed_f8, scale)
        T.ptx["cvt.rn.satfinite.scaled::n2::ue8m0.bf16x2.e4m3x2"](
            scaled[2, index], packed_f8, scale
        )
        T.ptx["cvt.rn.relu.satfinite.scaled::n2::ue8m0.bf16x2.e4m3x2"](
            scaled[3, index], packed_f8, scale
        )

        # cvt.rn{.relu}{.satfinite}.scaled::n2::ue8m0.bf16x2.e5m2x2
        T.ptx["cvt.rn.scaled::n2::ue8m0.bf16x2.e5m2x2"](scaled[4, index], packed_f8, scale)
        T.ptx["cvt.rn.relu.scaled::n2::ue8m0.bf16x2.e5m2x2"](scaled[5, index], packed_f8, scale)
        T.ptx["cvt.rn.satfinite.scaled::n2::ue8m0.bf16x2.e5m2x2"](
            scaled[6, index], packed_f8, scale
        )
        T.ptx["cvt.rn.relu.satfinite.scaled::n2::ue8m0.bf16x2.e5m2x2"](
            scaled[7, index], packed_f8, scale
        )

        # cvt.rn{.relu}{.satfinite}.scaled::n2::ue8m0.bf16x2.e2m1x2
        T.ptx["cvt.rn.scaled::n2::ue8m0.bf16x2.e2m1x2"](scaled[8, index], packed_f4, scale)
        T.ptx["cvt.rn.relu.scaled::n2::ue8m0.bf16x2.e2m1x2"](scaled[9, index], packed_f4, scale)
        T.ptx["cvt.rn.satfinite.scaled::n2::ue8m0.bf16x2.e2m1x2"](
            scaled[10, index], packed_f4, scale
        )
        T.ptx["cvt.rn.relu.satfinite.scaled::n2::ue8m0.bf16x2.e2m1x2"](
            scaled[11, index], packed_f4, scale
        )


SM100_OUTPUTS = (
    "pack_f4",
    "unpack_f4",
    "stochastic_f8",
    "stochastic_f4",
    "exponent",
    "from_exponent",
)
BF16X2_OUTPUTS = ("pack_f4", "unpack_f4", "scaled")

#: golden key -> (kernel name, output buffer, row) written above.
FORM_SLOTS: Mapping[str, tuple[str, str, int]] = {
    "e2m1x2.f32": ("sm100", "pack_f4", 0),
    "e2m1x2.f32.relu": ("sm100", "pack_f4", 1),
    "e2m1x2.f16x2": ("sm100", "pack_f4", 2),
    "e2m1x2.f16x2.relu": ("sm100", "pack_f4", 3),
    "f16x2.e2m1x2": ("sm100", "unpack_f4", 0),
    "f16x2.e2m1x2.relu": ("sm100", "unpack_f4", 1),
    "e4m3x4.f32.rs": ("sm100", "stochastic_f8", 0),
    "e4m3x4.f32.rs.relu": ("sm100", "stochastic_f8", 1),
    "e5m2x4.f32.rs": ("sm100", "stochastic_f8", 2),
    "e5m2x4.f32.rs.relu": ("sm100", "stochastic_f8", 3),
    "e2m1x4.f32.rs": ("sm100", "stochastic_f4", 0),
    "e2m1x4.f32.rs.relu": ("sm100", "stochastic_f4", 1),
    "ue8m0x2.f32.rz": ("sm100", "exponent", 0),
    "ue8m0x2.f32.rz.satfinite": ("sm100", "exponent", 1),
    "ue8m0x2.f32.rp": ("sm100", "exponent", 2),
    "ue8m0x2.f32.rp.satfinite": ("sm100", "exponent", 3),
    "ue8m0x2.bf16x2.rz": ("sm100", "exponent", 4),
    "ue8m0x2.bf16x2.rz.satfinite": ("sm100", "exponent", 5),
    "ue8m0x2.bf16x2.rp": ("sm100", "exponent", 6),
    "ue8m0x2.bf16x2.rp.satfinite": ("sm100", "exponent", 7),
    "bf16x2.ue8m0x2": ("sm100", "from_exponent", 0),
    "e2m1x2.bf16x2": ("bf16x2", "pack_f4", 0),
    "e2m1x2.bf16x2.relu": ("bf16x2", "pack_f4", 1),
    "bf16x2.e2m1x2": ("bf16x2", "unpack_f4", 0),
    "bf16x2.e2m1x2.relu": ("bf16x2", "unpack_f4", 1),
    "bf16x2.e2m1x2.satfinite": ("bf16x2", "unpack_f4", 2),
    "bf16x2.e2m1x2.relu.satfinite": ("bf16x2", "unpack_f4", 3),
    "bf16x2.e4m3x2.scaled": ("bf16x2", "scaled", 0),
    "bf16x2.e4m3x2.scaled.relu": ("bf16x2", "scaled", 1),
    "bf16x2.e4m3x2.scaled.satfinite": ("bf16x2", "scaled", 2),
    "bf16x2.e4m3x2.scaled.relu.satfinite": ("bf16x2", "scaled", 3),
    "bf16x2.e5m2x2.scaled": ("bf16x2", "scaled", 4),
    "bf16x2.e5m2x2.scaled.relu": ("bf16x2", "scaled", 5),
    "bf16x2.e5m2x2.scaled.satfinite": ("bf16x2", "scaled", 6),
    "bf16x2.e5m2x2.scaled.relu.satfinite": ("bf16x2", "scaled", 7),
    "bf16x2.e2m1x2.scaled": ("bf16x2", "scaled", 8),
    "bf16x2.e2m1x2.scaled.relu": ("bf16x2", "scaled", 9),
    "bf16x2.e2m1x2.scaled.satfinite": ("bf16x2", "scaled", 10),
    "bf16x2.e2m1x2.scaled.relu.satfinite": ("bf16x2", "scaled", 11),
}

if set(FORM_SLOTS) != set(GOLDENS):
    raise AssertionError("kernel form slots and GPU goldens disagree")


def make_sm100_arguments() -> dict[str, Any]:
    return {
        "f32_source": F32_SOURCE.view(np.float32).copy(),
        "f16x2_source": F16X2_SOURCE.copy(),
        "bf16x2_source": BF16X2_SOURCE.copy(),
        "f4x2_source": F4X2_SOURCE.astype(np.uint16),
        "ue8x2_source": UE8X2_SOURCE.astype(np.uint16),
        "rbits_source": RBITS_SOURCE.copy(),
        "pack_f4": np.zeros((4, VALUE_COUNT), dtype=np.uint8),
        "unpack_f4": np.zeros((2, VALUE_COUNT), dtype=np.uint32),
        "stochastic_f8": np.zeros((4, VALUE_COUNT), dtype=np.uint32),
        "stochastic_f4": np.zeros((2, VALUE_COUNT), dtype=np.uint16),
        "exponent": np.zeros((8, VALUE_COUNT), dtype=np.uint16),
        "from_exponent": np.zeros((1, VALUE_COUNT), dtype=np.uint32),
    }


def make_bf16x2_arguments() -> dict[str, Any]:
    return {
        "bf16x2_source": BF16X2_SOURCE.copy(),
        "f4x2_source": F4X2_SOURCE.astype(np.uint16),
        "f8x2_source": F8X2_SOURCE.astype(np.uint16),
        "scale_source": SCALE_SOURCE.astype(np.uint16),
        "pack_f4": np.zeros((2, VALUE_COUNT), dtype=np.uint8),
        "unpack_f4": np.zeros((4, VALUE_COUNT), dtype=np.uint32),
        "scaled": np.zeros((12, VALUE_COUNT), dtype=np.uint32),
    }


KERNELS = {
    "sm100": (narrow_cvt_sm100_forms, make_sm100_arguments, SM100_OUTPUTS),
    "bf16x2": (narrow_cvt_bf16x2_forms, make_bf16x2_arguments, BF16X2_OUTPUTS),
}


__all__ = [
    "BF16X2_OUTPUTS",
    "FORM_SLOTS",
    "KERNELS",
    "SM100_OUTPUTS",
    "VALUE_COUNT",
    "make_bf16x2_arguments",
    "make_sm100_arguments",
    "narrow_cvt_bf16x2_forms",
    "narrow_cvt_sm100_forms",
]
