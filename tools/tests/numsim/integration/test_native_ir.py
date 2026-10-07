"""Native node access preserves NumSim's identity and metadata contracts."""


import json

import pytest
import tvm
import tvm_ffi
from tvm import tirx
from tvm.ir import Expr
from tvm.script import tirx as T
from tvm.tirx import Stmt

from tirx_harness.numsim.transpiler import frontend, host_prelude, native_frontend
from tirx_harness.numsim.transpiler.ptx_dialect import PTX_SINK, PtxCallDecodeError, decode_ptx_call
from tirx_harness.numsim.transpiler.semantic_ir import semantic_ir_json
from tests.numsim.integration.test_host_prelude import host_encoded_tensor_map_store
from tests.numsim.support.kernels import lane_add


def _substitute_vars(value, replacements):
    return tvm_ffi.structural_map(
        value, (tirx.Var, lambda var: replacements.get(var, var)), order="post"
    )


@pytest.fixture
def native_calls(monkeypatch):
    library = native_frontend._library()
    calls = []

    class TrackedLibrary:
        def __getitem__(self, name):
            def call(*args):
                calls.append((name, args))
                return library[name](*args)

            return call

    monkeypatch.setattr(native_frontend, "_library", lambda: TrackedLibrary())
    return calls


def test_native_post_order_preserves_dag_order_handles_and_spans(native_calls):
    span = tvm.ir.SequentialSpan([
        tvm.ir.Span(tvm.ir.SourceName("first.py"), 3, 3, 1, 8),
        tvm.ir.Span(tvm.ir.SourceName("second.py"), 7, 7, 2, 9),
    ])
    variable = tirx.Var("value", "int32", span)
    repeated = tirx.Evaluate(variable + 1, span=span)
    body = tirx.SeqStmt([repeated, repeated, tirx.Evaluate(variable + 2)])
    before = tvm.ir.save_json(body)
    last = body.seq[2]
    expected = [
        variable, repeated.value.b, repeated.value, repeated, last.value.b, last.value, last, body
    ]
    actual = native_frontend.post_order_nodes(body)
    assert len(actual) == len(expected)
    assert all(left.same_as(right) for left, right in zip(actual, expected, strict=True))
    assert sum(node.same_as(repeated) for node in actual) == 1
    assert next(node for node in actual if node.same_as(repeated)).span.same_as(span)
    assert tvm.ir.save_json(body) == before
    assert [name for name, _ in native_calls] == ["numsim_post_order_nodes"]


@pytest.mark.parametrize("root_span", [False, True])
@pytest.mark.parametrize("sequential_span", [False, True])
def test_semantic_serialization_removes_spans_without_changing_the_ir(root_span, sequential_span):
    span = tvm.ir.Span(tvm.ir.SourceName("kernel.py"), 3, 3, 1, 8)
    if sequential_span:
        span = tvm.ir.SequentialSpan([
            span,
            tvm.ir.Span(tvm.ir.SourceName("inlined.py"), 7, 7, 2, 9),
        ])
    variable = tirx.Var("value", "int32", span)
    shared = tirx.Evaluate(variable + 1, span=span)
    body = tirx.SeqStmt(
        [shared, shared, tirx.Evaluate(variable + 2)], span=span if root_span else None
    )
    before = tvm.ir.save_json(body)

    expected_variable = tirx.Var("value", "int32")
    expected_shared = tirx.Evaluate(expected_variable + 1)
    expected = tirx.SeqStmt([
        expected_shared, expected_shared, tirx.Evaluate(expected_variable + 2)
    ])
    serialized = semantic_ir_json(body)
    assert serialized == tvm.ir.save_json(expected)
    restored = tvm.ir.load_json(serialized)
    assert tvm_ffi.structural_equal(restored, expected, map_free_vars=True)
    assert restored.seq[0].same_as(restored.seq[1])
    assert tvm.ir.save_json(body) == before


def test_frontend_source_map_consumes_one_native_body_walk(native_calls):
    spec = frontend.analyze(lane_add).kernels[0]
    # The native analyzer owns the whole walk: one analysis call, no Python walk.
    assert sum(
        name == "numsim_analyze_module" and args[0][0].same_as(lane_add)
        for name, args in native_calls
    ) == 1
    assert not any(name == "numsim_post_order_nodes" for name, _ in native_calls)
    expected = native_frontend.post_order_nodes(lane_add.body)
    assert len(spec.source_map) == len(expected)
    for entry, node in zip(spec.source_map, expected, strict=True):
        assert entry.node.same_as(node)
        assert entry.kind == type(node).__name__


@pytest.mark.parametrize("rename", [False, True])
def test_host_promotion_preserves_parameters_types_spans_and_references(native_calls, rename):
    original = host_encoded_tensor_map_store
    allocation = original.body.seq[0].var
    span = tvm.ir.SequentialSpan([
        tvm.ir.Span(tvm.ir.SourceName("host.py"), 4, 4, 1, 9),
        tvm.ir.Span(tvm.ir.SourceName("device.py"), 8, 8, 2, 10),
    ])
    variable = tirx.Var("output" if rename else str(allocation.name), allocation.ty, span)
    body = tirx.SeqStmt([
        tirx.Bind(variable, original.body.seq[0].value),
        *(_substitute_vars(stmt, {allocation: variable}) for stmt in original.body.seq[1:]),
    ])
    func = tirx.PrimFunc(original.params, body, original.ret_type, original.attrs, span)
    before = tvm.ir.save_json(func)
    normalized = host_prelude.normalize_host_tensor_map_prelude(func)
    promoted = normalized.params[-1]
    assert normalized.params[0].same_as(func.params[0])
    assert promoted.same_as(variable) == (not rename)
    assert promoted.ty.same_as(variable.ty) and promoted.span.same_as(span)
    assert normalized.span.same_as(span) and normalized.ret_type.same_as(func.ret_type)
    metadata = json.loads(str(normalized.attrs["numsim.implicit_tensor_maps"]))
    assert metadata[0]["name"] == str(promoted.name)
    assert str(promoted.name) == ("output_tmap" if rename else str(variable.name))
    expected_body = _substitute_vars(body.seq[-1], {variable: promoted})
    expected = tirx.PrimFunc(
        [*func.params, promoted], expected_body, func.ret_type, normalized.attrs, span
    )
    assert tvm_ffi.structural_equal(normalized, expected)
    assert tvm_ffi.structural_equal(normalized.ty, expected.ty)
    assert any(
        node.same_as(promoted) for node in native_frontend.post_order_nodes(normalized.body)
    )
    assert tvm.ir.save_json(func) == before
    assert sum(name == "numsim_normalize_host_tensor_maps" for name, _ in native_calls) == 1


@T.prim_func
def native_ptx_metadata():
    T.device_entry()
    hi = T.local_scalar("uint32")
    packed = T.local_scalar("uint64")
    T.ptx.mov.b64(T.ptx.SINK, hi, packed)


@pytest.fixture
def ptx_call():
    calls = []

    def visit(node):
        if isinstance(node, tvm.ir.Call) and str(node.op.name).startswith("tirx.ptx."):
            calls.append(node)

    tvm_ffi.structural_walk(native_ptx_metadata.body, ((Expr, Stmt), visit))
    assert len(calls) == 1
    return calls[0]


def test_ptx_decoder_consumes_native_parts_without_copying_operands(native_calls, ptx_call):
    before = tvm.ir.save_json(ptx_call)
    decoded = decode_ptx_call(ptx_call)
    assert decoded.operand("d")[0] is PTX_SINK
    assert decoded.operand("d")[1].same_as(ptx_call.args[0])
    assert decoded.scalar_operand("a").same_as(ptx_call.args[1])
    assert decoded.span.same_as(ptx_call.span) and decoded.result_type.same_as(ptx_call.ty)
    assert tvm.ir.save_json(ptx_call) == before
    assert [name for name, _ in native_calls] == ["numsim_ptx_call_parts"]


@pytest.mark.parametrize("position", [-1, -2])
def test_native_ptx_metadata_keeps_typed_errors_and_source_span(native_calls, ptx_call, position):
    args = list(ptx_call.args)
    args[position] = tirx.IntImm("int32", 0)
    invalid = tvm.ir.Call(ptx_call.op, args, span=ptx_call.span, ret_ty=ptx_call.ty)
    with pytest.raises(PtxCallDecodeError, match="must be a StringImm, got IntImm") as caught:
        decode_ptx_call(invalid)
    assert caught.value.source_span.same_as(ptx_call.span)
    assert caught.value.unsupported == (str(ptx_call.op.name),)
    assert [name for name, _ in native_calls] == ["numsim_ptx_call_parts"]
