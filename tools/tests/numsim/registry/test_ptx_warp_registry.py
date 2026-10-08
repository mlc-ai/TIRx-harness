from __future__ import annotations

from tvm.ir import Expr
from tvm.tirx import Stmt
from tvm_ffi import structural_walk

from tests.numsim.runtime.test_ptx_warp_collectives import (
    raw_ptx_match_float_carriers,
    raw_ptx_match_redux_and_activemask,
    raw_ptx_vote_predicate_carrier,
)
from tests.numsim.support.manifest import call_op_names, emitted_calls, kernel_manifest
from tirx_harness.numsim.transpiler import native_frontend
from tirx_harness.numsim.transpiler.ptx_dialect import PTX_SCHEMA_BY_OP_NAME, decode_ptx_call

ACTIVEMASK = "tirx.ptx.activemask"
MATCH_ANY = "tirx.ptx.match_any_sync"
MATCH_ALL = "tirx.ptx.match_all_sync"
MATCH_ALL_P = "tirx.ptx.match_all_sync_p"
REDUX_BITWISE = "tirx.ptx.redux_sync_bitwise"
VOTE_BALLOT = "tirx.ptx.vote_sync_ballot"

_COVERED_WARP_CALLS = (ACTIVEMASK, MATCH_ANY, MATCH_ALL, MATCH_ALL_P, REDUX_BITWISE, VOTE_BALLOT)
_MATCH_CALLS = frozenset({MATCH_ANY, MATCH_ALL, MATCH_ALL_P})


def _warp_calls():
    calls = []

    def visit(node):
        if str(getattr(getattr(node, "op", None), "name", "")).startswith("tirx.ptx."):
            calls.append((func, manifest, node))

    for func in (
        raw_ptx_match_redux_and_activemask,
        raw_ptx_match_float_carriers,
        raw_ptx_vote_predicate_carrier,
    ):
        manifest = kernel_manifest(func)
        structural_walk(func.body, ((Expr, Stmt), visit))
    return calls


def test_covered_warp_schema_boundary_is_register_only_and_has_no_memory_effect():
    for name in _COVERED_WARP_CALLS:
        entry = PTX_SCHEMA_BY_OP_NAME[name]
        assert not entry.orders_memory
        assert entry.operands
        assert {operand.kind for operand in entry.operands} == {"reg"}


def test_runtime_warp_forms_resolve_with_their_operand_and_result_contracts():
    seen = set()
    specs = {spec["ir_name"]: spec for spec in native_frontend.registry_ops()}
    for func, kernel, call in _warp_calls():
        if str(call.op.name) not in _COVERED_WARP_CALLS:
            continue
        decoded = decode_ptx_call(call)
        seen.add(decoded.op_name)
        spec = specs[decoded.op_name]
        assert decoded.op_name in call_op_names(kernel)
        assert spec["support"] == "modeled"
        assert spec["family"] == (
            "warp_query" if decoded.op_name == ACTIVEMASK else "warp_collective"
        )
        instruction = emitted_calls(func, call)[0]
        if decoded.op_name == ACTIVEMASK:
            assert instruction.head == "v2::warp::activemask"
        elif decoded.op_name in _MATCH_CALLS:
            assert instruction.function == "v2::warp::match_sync"
            assert instruction.generics.startswith("v2::warp::variant::Match<")
        elif decoded.op_name == REDUX_BITWISE:
            assert instruction.function == "v2::warp::redux_sync"
            assert instruction.generics in {
                f"v2::warp::variant::Redux<v2::reg::variant::B32, v2::warp::variant::{operation}>"
                for operation in ("And", "Or", "Xor")
            }
        else:
            assert instruction.head == "v2::warp::vote_sync::<v2::warp::variant::Ballot>"
    assert seen == set(_COVERED_WARP_CALLS)
