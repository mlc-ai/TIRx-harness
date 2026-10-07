"""Focused NumSim semantics for `atom{.sem}{.scope}{.space}.cas.type`."""

from __future__ import annotations

import numpy as np
from tvm.script import tirx as T

from tirx_harness.numsim import Engine, transpile


@T.prim_func
def cas_claim_shared(out: T.Buffer((4,), "uint32")):
    """One lane exercises successful and failed compare-and-swap operations."""

    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    slot = T.alloc_buffer((1,), "uint32", scope="shared")
    if lane == 0:
        slot[0] = T.uint32(0)
    T.cuda.cta_sync()
    if lane == 0:
        old = T.alloc_local((1,), "uint32")
        T.ptx.atom.relaxed.cta.shared.cas.b32(
            old[0], slot.ptr_to([0]), T.uint32(0), T.uint32(1)
        )
        out[0] = old[0]
        out[1] = slot[0]
        T.ptx.atom.relaxed.cta.shared.cas.b32(
            old[0], slot.ptr_to([0]), T.int32(0), T.int32(-1)
        )
        out[2] = old[0]
        out[3] = slot[0]


@T.prim_func
def cas_claim_shared_b16(out: T.Buffer((2,), "uint16")):
    """Exercise the 16-bit PTX form with signed bit-carrier operands."""

    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    slot = T.alloc_buffer((1,), "uint16", scope="shared")
    if lane == 0:
        slot[0] = T.uint16(0xFFFF)
    T.cuda.cta_sync()
    if lane == 0:
        old = T.alloc_local((1,), "uint16")
        T.ptx.atom.relaxed.cta.shared.cas.b16(
            old[0], slot.ptr_to([0]), T.int16(-1), T.int16(7)
        )
        out[0] = old[0]
        out[1] = slot[0]


@T.prim_func
def cas_claim_global_b64(cell: T.Buffer((1,), "uint64"), out: T.Buffer((2,), "uint64")):
    """Exercise the 64-bit PTX form on global memory."""

    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    if lane == 0:
        old = T.alloc_local((1,), "uint64")
        T.ptx.atom.relaxed.gpu.global_.cas.b64(
            old[0], cell.ptr_to([0]), T.uint64(7), T.uint64(9)
        )
        out[0] = old[0]
        out[1] = cell[0]


def test_atom_cas_returns_the_prior_value_and_swaps():
    module = transpile(cas_claim_shared)
    result = Engine(max_workers=1).run(
        module, {"out": np.zeros(4, dtype=np.uint32)}, outputs=("out",)
    )
    # `.cas` always writes the old value to its destination, and only installs
    # the replacement when the bitwise comparison succeeds.
    np.testing.assert_array_equal(result.outputs["out"], np.array([0, 1, 1, 1]))


def test_atom_cas_supports_b16_signed_carriers():
    module = transpile(cas_claim_shared_b16)
    result = Engine(max_workers=1).run(
        module, {"out": np.zeros(2, dtype=np.uint16)}, outputs=("out",)
    )
    np.testing.assert_array_equal(result.outputs["out"], np.array([0xFFFF, 7]))


def test_atom_cas_supports_global_b64():
    module = transpile(cas_claim_global_b64)
    result = Engine(max_workers=1).run(
        module,
        {
            "cell": np.array([7], dtype=np.uint64),
            "out": np.zeros(2, dtype=np.uint64),
        },
        outputs=("cell", "out"),
    )
    np.testing.assert_array_equal(result.outputs["cell"], np.array([9]))
    np.testing.assert_array_equal(result.outputs["out"], np.array([7, 9]))
