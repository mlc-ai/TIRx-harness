from __future__ import annotations

from tvm.ir import Expr
from tvm.script import tirx as T
from tvm.tirx import Stmt
from tvm_ffi import structural_walk

from tests.numsim.support.manifest import emitted_calls
from tirx_harness.numsim.transpiler import native_frontend


@T.prim_func
def all_lop3_shapes(
    a: T.Buffer((1,), "uint32"),
    b: T.Buffer((1,), "int32"),
    c: T.Buffer((1,), "float32"),
    d: T.Buffer((1,), "uint32"),
    p: T.Buffer((1,), "uint32"),
    q: T.Buffer((1,), "uint32"),
):
    T.device_entry()
    T.ptx.lop3.b32(d[0], a[0], b[0], c[0], 0)
    T.ptx.lop3.b32(d[0], a[0], b[0], c[0], 255)
    T.ptx.lop3.and_.b32(d[0], p[0], a[0], b[0], c[0], 0x40, T.ptx.pred(q[0]))
    T.ptx.lop3.or_.b32(d[0], p[0], a[0], b[0], c[0], 0xFE, T.ptx.pred(q[0]))
    T.ptx.lop3.and_.b32(p[0], a[0], b[0], c[0], 0x80, T.ptx.pred(q[0]))
    T.ptx.lop3.or_.b32(p[0], a[0], b[0], c[0], 0x1A, T.ptx.pred(q[0]))


def _lop3_calls():
    calls = []

    def visit(node: object) -> None:
        if type(node).__name__ == "Call" and str(getattr(node.op, "name", "")).startswith(
            "tirx.ptx.lop3"
        ):
            calls.append(node)

    structural_walk(all_lop3_shapes.body, ((Expr, Stmt), visit))
    return calls


def test_lop3_registry_owns_all_three_public_shapes():
    specs = {spec["ir_name"]: spec for spec in native_frontend.registry_ops()}
    for name in ("tirx.ptx.lop3", "tirx.ptx.lop3_bool", "tirx.ptx.lop3_bool_sink"):
        spec = specs[name]
        assert spec["support"] == "modeled"
        assert spec["family"] == "register_truth_table_logic"


def test_lop3_truth_tables_and_bool_shapes_have_closed_specializations():
    observed = {
        (str(call.op.name), emitted_calls(all_lop3_shapes, call)[0].head) for call in _lop3_calls()
    }

    assert observed == {
        ("tirx.ptx.lop3", "v2::reg::lop3::<v2::reg::variant::Lop3<0>>"),
        ("tirx.ptx.lop3", "v2::reg::lop3::<v2::reg::variant::Lop3<255>>"),
        (
            "tirx.ptx.lop3_bool",
            "v2::reg::lop3::<v2::reg::variant::Lop3Bool<64, v2::reg::variant::BoolAnd>>",
        ),
        (
            "tirx.ptx.lop3_bool",
            "v2::reg::lop3::<v2::reg::variant::Lop3Bool<254, v2::reg::variant::BoolOr>>",
        ),
        (
            "tirx.ptx.lop3_bool_sink",
            "v2::reg::lop3::<v2::reg::variant::Lop3Bool<128, v2::reg::variant::BoolAnd>>",
        ),
        (
            "tirx.ptx.lop3_bool_sink",
            "v2::reg::lop3::<v2::reg::variant::Lop3Bool<26, v2::reg::variant::BoolOr>>",
        ),
    }
