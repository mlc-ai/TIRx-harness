"""Engine instructions emitted for public packed PTX ``cvt`` calls."""

from __future__ import annotations

import numpy as np
import pytest
import tvm
from tvm.ir import Expr
from tvm.script import tirx as T
from tvm.tirx import Stmt
from tvm_ffi import structural_walk

from tirx_harness import numsim
from tests.numsim.support.manifest import emitted_calls


def _cvt_call(spelling: str, destination_dtype: str, *arguments: str):
    func = tvm.script.from_source(
        f"""
@T.prim_func
def kernel():
    T.device_entry()
    destination = T.local_scalar("{destination_dtype}")
    T.ptx["{spelling}"](destination, {", ".join(arguments)})
""",
        {"T": T},
    )
    calls = []

    def visit(node: object) -> None:
        name = str(getattr(getattr(node, "op", None), "name", ""))
        if type(node).__name__ == "Call" and (
            name == "tirx.ptx.cvt" or name.startswith("tirx.ptx.cvt_")
        ):
            calls.append(node)

    structural_walk(func.body, ((Expr, Stmt), visit))
    assert len(calls) == 1
    return func, calls[0]


def _emitted_cvt(spelling: str, destination_dtype: str, *arguments: str) -> str:
    """The head of the engine instruction the call emits."""

    func, call = _cvt_call(spelling, destination_dtype, *arguments)
    (emitted,) = emitted_calls(func, str(call.op.name))
    return emitted.head


def _packed_head(source: str, destination: str, *modes: str) -> str:
    """The ``v2::reg::cvt`` head of a packed conversion with ``modes``."""

    variant = "v2::reg::variant::"
    spelled = ", ".join(variant + mode for mode in modes)
    return (
        f"v2::reg::cvt::<{variant}Cvt<{variant}{source}, {variant}{destination}, "
        f"{variant}PackedMode<{spelled}>>>"
    )


def test_relu_and_satfinite_reach_the_emitted_conversion():
    argument = "T.uint16(0x3840)"
    plain = _emitted_cvt("cvt.rn.bf16x2.e4m3x2", "uint32", argument)
    relu = _emitted_cvt("cvt.rn.relu.bf16x2.e4m3x2", "uint32", argument)
    saturating = _emitted_cvt("cvt.rn.satfinite.bf16x2.e4m3x2", "uint32", argument)
    both = _emitted_cvt("cvt.rn.relu.satfinite.bf16x2.e4m3x2", "uint32", argument)
    scaled = _emitted_cvt(
        "cvt.rn.scaled::n2::ue8m0.bf16x2.e4m3x2",
        "uint32",
        argument,
        "T.uint16(127)",
    )

    assert plain == _packed_head("E4m3x2", "Bf16x2", "Rn", "NoSatFinite", "NoRelu")
    assert relu == _packed_head("E4m3x2", "Bf16x2", "Rn", "NoSatFinite", "Relu")
    assert saturating == _packed_head("E4m3x2", "Bf16x2", "Rn", "SatFinite", "NoRelu")
    assert both == _packed_head("E4m3x2", "Bf16x2", "Rn", "SatFinite", "Relu")
    assert scaled == _packed_head(
        "E4m3x2", "Bf16x2", "Rn", "NoSatFinite", "NoRelu", "ScaledUe8m0N2"
    )
    assert len({plain, relu, saturating, both, scaled}) == 5


def test_previously_unmodeled_neighbors_preserve_exact_modifiers():
    pair = ("T.float32(1.0)", "T.float32(2.0)")
    for spelling, dtype, arguments, expected in (
        (
            "cvt.rs.f16x2.f32",
            "uint32",
            (*pair, "T.uint32(0)"),
            _packed_head("F32", "F16x2", "Rs", "NoSatFinite", "NoRelu"),
        ),
        (
            "cvt.rs.satfinite.bf16x2.f32",
            "uint32",
            (*pair, "T.uint32(0)"),
            _packed_head("F32", "Bf16x2", "Rs", "SatFinite", "NoRelu"),
        ),
        (
            "cvt.rn.satfinite.e2m3x2.f32",
            "uint16",
            pair,
            _packed_head("F32", "E2m3x2", "Rn", "SatFinite", "NoRelu"),
        ),
        (
            "cvt.rs.satfinite.e3m2x4.f32",
            "uint32",
            (*pair, "T.float32(3.0)", "T.float32(4.0)", "T.uint32(0)"),
            _packed_head("F32", "E3m2x4", "Rs", "SatFinite", "NoRelu"),
        ),
        (
            "cvt.rn.scaled::n2::ue8m0.bf16x2.s2f6x2",
            "uint32",
            ("T.uint16(1)", "T.uint16(127)"),
            _packed_head("S2f6x2", "Bf16x2", "Rn", "NoSatFinite", "NoRelu", "ScaledUe8m0N2"),
        ),
    ):
        assert _emitted_cvt(spelling, dtype, *arguments) == expected, spelling


def test_f6_form_transpiles_and_preserves_exact_half_bits(tmp_path):
    @T.prim_func
    def f6_cvt(source: T.Buffer((32,), "uint16"), output: T.Buffer((32,), "uint32")):
        T.device_entry()
        _warp = T.warp_id([1])
        lane = T.lane_id([32])
        T.ptx["cvt.rn.f16x2.e2m3x2"](output[lane], source[lane])

    # E2M3 encodes 1, 2 and 4 as 0x08, 0x10 and 0x18; sign is bit5.
    source = np.tile(np.array([0, 0x0810, 0x2018, 0x2808], np.uint16), 8)
    expected = np.tile(np.array([0, 0x3C004000, 0x80004400, 0xBC003C00], np.uint32), 8)
    result = numsim.Engine().run(
        numsim.transpile(f6_cvt, cache_dir=tmp_path),
        {"source": source, "output": np.zeros(32, np.uint32)},
    )
    np.testing.assert_array_equal(result.outputs["output"], expected)


def test_stochastic_and_exponent_forms_record_their_rounding():
    stochastic = _emitted_cvt(
        "cvt.rs.relu.satfinite.e2m1x4.f32",
        "uint16",
        "T.float32(1.0)",
        "T.float32(2.0)",
        "T.float32(3.0)",
        "T.float32(4.0)",
        "T.uint32(0)",
    )
    assert stochastic == _packed_head("F32", "E2m1x4", "Rs", "SatFinite", "Relu")
    exponent = _emitted_cvt(
        "cvt.rp.satfinite.ue8m0x2.f32",
        "uint16",
        "T.float32(1.0)",
        "T.float32(2.0)",
    )
    assert exponent == _packed_head("F32", "Ue8m0x2", "Rp", "SatFinite", "NoRelu")
    plain = _emitted_cvt(
        "cvt.rz.ue8m0x2.f32",
        "uint16",
        "T.float32(1.0)",
        "T.float32(2.0)",
    )
    assert plain == _packed_head("F32", "Ue8m0x2", "Rz", "NoSatFinite", "NoRelu")


@pytest.mark.parametrize(
    ("spelling", "destination_dtype", "arguments"),
    [
        pytest.param(
            "cvt.rn.satfinite.ftz.e4m3x2.f16x2",
            "uint16",
            ("T.uint32(0x3C003C00)",),
            id="ftz_e4m3x2_from_f16x2",
        ),
        pytest.param(
            "cvt.rn.satfinite.ftz.e2m1x2.f32",
            "uint8",
            ("T.float32(1.0)", "T.float32(2.0)"),
            id="ftz_e2m1x2_from_f32",
        ),
        pytest.param(
            "cvt.rz.f32.f32",
            "float32",
            ("T.float32(1.5)",),
            id="directed_rounding_scalar",
        ),
    ],
)
def test_non_table_cvt_neighbours_are_rejected_by_target_parser(
    spelling, destination_dtype, arguments
):
    with pytest.raises(tvm.error.DiagnosticError):
        _cvt_call(spelling, destination_dtype, *arguments)
