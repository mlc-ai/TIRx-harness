from __future__ import annotations

import numpy as np
import tvm
from tvm.ir import Expr
from tvm.script import tirx as T
from tvm.tirx import Stmt
from tvm_ffi import structural_walk

from tirx_harness import numsim
from tests.numsim.support.manifest import emitted_calls

_PZO_CALL_IDS = {
    "f16": "tirx.ptx.cvt_pzo_scalar_f32",
    "bf16": "tirx.ptx.cvt_pzo_scalar_f32",
    "f16x2": "tirx.ptx.cvt_pzo_fp16x2_f32",
    "bf16x2": "tirx.ptx.cvt_pzo_fp16x2_f32",
    "tf32": "tirx.ptx.cvt_pzo_tf32_f32",
}

_PZO_DESTINATION_MARKERS = {
    "f16": "F16",
    "bf16": "Bf16",
    "f16x2": "F16x2",
    "bf16x2": "Bf16x2",
    "tf32": "Tf32",
}


def _pzo_spelling(destination: str, rounding: str, relu: bool, satfinite: bool) -> str:
    modifiers = ["cvt", rounding]
    if destination == "tf32":
        if satfinite:
            modifiers.append("satfinite")
        if relu:
            modifiers.append("relu")
    else:
        if relu:
            modifiers.append("relu")
        if satfinite:
            modifiers.append("satfinite")
    modifiers.extend(("pzo", destination, "f32"))
    return ".".join(modifiers)


def _pzo_head(destination: str, rounding: str, relu: bool, satfinite: bool) -> str:
    """The head of the engine instruction one ``pzo`` conversion emits."""

    variant = "v2::reg::variant::"
    modes = (
        rounding.title(),
        "SatFinite" if satfinite else "NoSatFinite",
        "Relu" if relu else "NoRelu",
        "NoScale",
        "Pzo",
    )
    spelled = ", ".join(variant + mode for mode in modes)
    return (
        f"v2::reg::cvt::<{variant}Cvt<{variant}F32, "
        f"{variant}{_PZO_DESTINATION_MARKERS[destination]}, "
        f"{variant}PackedMode<{spelled}>>>"
    )


def _decoded_call(spelling: str, destination: str):
    destination_dtype = "uint16" if destination in {"f16", "bf16"} else "uint32"
    arguments = "T.float32(-0.0)"
    if destination.endswith("x2"):
        arguments += ", T.float32(1.0)"
    function = tvm.script.from_source(
        f"""
@T.prim_func
def kernel():
    T.device_entry()
    output = T.local_scalar("{destination_dtype}")
    T.ptx["{spelling}"](output, {arguments})
""",
        {"T": T},
    )
    calls = []

    def visit(node: object) -> None:
        name = str(getattr(getattr(node, "op", None), "name", ""))
        if type(node).__name__ == "Call" and name.startswith("tirx.ptx.cvt_pzo_"):
            calls.append(node)

    structural_walk(function.body, ((Expr, Stmt), visit))
    assert len(calls) == 1
    return function, calls[0]


def test_all_forty_pzo_forms_resolve_to_distinct_reviewed_semantics():
    seen = set()
    for destination in ("f16", "bf16", "f16x2", "bf16x2", "tf32"):
        for rounding in ("rn", "rz"):
            for relu in (False, True):
                for satfinite in (False, True):
                    spelling = _pzo_spelling(destination, rounding, relu, satfinite)
                    function, call = _decoded_call(spelling, destination)
                    assert call.op.name == _PZO_CALL_IDS[destination]
                    (emitted,) = emitted_calls(function, str(call.op.name))
                    assert emitted.head == _pzo_head(destination, rounding, relu, satfinite)
                    seen.add(emitted.head)
    assert len(seen) == 40


@T.prim_func
def pzo_conversions(
    source: T.Buffer((32,), "float32"),
    low: T.Buffer((32,), "float32"),
    f16: T.Buffer((32,), "uint16"),
    bf16: T.Buffer((32,), "uint16"),
    tf32: T.Buffer((32,), "uint32"),
    packed_f16: T.Buffer((32,), "uint32"),
    packed_bf16: T.Buffer((32,), "uint32"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    T.ptx["cvt.rz.pzo.f16.f32"](f16[lane], source[lane])
    T.ptx["cvt.rz.pzo.bf16.f32"](bf16[lane], source[lane])
    T.ptx["cvt.rz.pzo.tf32.f32"](tf32[lane], source[lane])
    T.ptx["cvt.rz.pzo.f16x2.f32"](packed_f16[lane], source[lane], low[lane])
    T.ptx["cvt.rz.pzo.bf16x2.f32"](packed_bf16[lane], source[lane], low[lane])


def test_pzo_is_post_conversion_and_applies_to_each_packed_result(tmp_path):
    tiny = np.asarray([0x8000_0001], np.uint32).view(np.float32)[0]
    source = np.tile(
        np.asarray([-0.0, tiny, -1.0, 0.0, 1.0, -np.inf, np.inf, np.nan], np.float32),
        4,
    )
    low = np.roll(source, 1)
    arguments = {
        "source": source,
        "low": low,
        "f16": np.zeros(32, np.uint16),
        "bf16": np.zeros(32, np.uint16),
        "tf32": np.zeros(32, np.uint32),
        "packed_f16": np.zeros(32, np.uint32),
        "packed_bf16": np.zeros(32, np.uint32),
    }
    result = numsim.Engine().run(
        numsim.transpile(pzo_conversions, cache_dir=tmp_path),
        arguments,
    )
    expected_f16 = np.tile(
        np.asarray([0, 0, 0xBC00, 0, 0x3C00, 0xFC00, 0x7C00, 0x7FFF], np.uint16),
        4,
    )
    expected_bf16 = np.tile(
        np.asarray([0, 0, 0xBF80, 0, 0x3F80, 0xFF80, 0x7F80, 0x7FFF], np.uint16),
        4,
    )
    expected_tf32 = np.tile(
        np.asarray(
            [0, 0, 0xBF80_0000, 0, 0x3F80_0000, 0xFF80_0000, 0x7F80_0000, 0x7FFF_E000],
            np.uint32,
        ),
        4,
    )
    expected = {
        "f16": expected_f16,
        "bf16": expected_bf16,
        "tf32": expected_tf32,
        "packed_f16": (expected_f16.astype(np.uint32) << 16) | np.roll(expected_f16, 1),
        "packed_bf16": (expected_bf16.astype(np.uint32) << 16) | np.roll(expected_bf16, 1),
    }
    for name, values in expected.items():
        np.testing.assert_array_equal(result.outputs[name], values, err_msg=name)
