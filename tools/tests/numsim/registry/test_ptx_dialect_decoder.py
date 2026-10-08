from __future__ import annotations

import pickle

import pytest
import tvm
from tvm import ir
from tvm.ir import Expr
from tvm.script import tirx as T
from tvm.tirx import Stmt
from tvm_ffi import structural_walk

from tests.numsim.support.manifest import call_op_names, resolved_kernel
from tirx_harness.numsim.transpiler.ptx_dialect import (
    PTX_SINK,
    DecodedPtxCall,
    PtxCallDecodeError,
    decode_ptx_call,
)


def _parse(source: str):
    return tvm.script.from_source(source, {"T": T})


def _calls(func, prefix: str = "tirx."):
    calls = []

    def visit(node):
        if isinstance(node, ir.Call) and str(getattr(node.op, "name", "")).startswith(prefix):
            calls.append(node)

    structural_walk(func.body, ((Expr, Stmt), visit))
    return calls


def _call(func, op_name: str):
    matches = [call for call in _calls(func) if str(call.op.name) == op_name]
    assert len(matches) == 1
    return matches[0]


def _with_args(call, args):
    return ir.Call(call.op, list(args), attrs=call.attrs, span=call.span, ret_ty=call.ty)


def test_decode_mbarrier_and_wait_from_public_dsl():
    func = _parse(
        """
@T.prim_func
def kernel(bar: T.handle):
    T.device_entry()
    T.ptx.mbarrier.init.shared.b64(bar, T.uint32(1))
    T.ptx.tcgen05.wait__ld.sync.aligned()
    T.cuda.cta_sync()
"""
    )

    init = decode_ptx_call(_call(func, "tirx.ptx.mbarrier_init"))
    assert init.op_name == "tirx.ptx.mbarrier_init"
    assert tuple(init.operands) == ("addr", "count")
    assert isinstance(init.scalar_operand("addr"), ir.Call)
    assert int(init.scalar_operand("count")) == 1
    assert dict(init.modifiers) == {
        "action": "init",
        "layout": "",
        "space": "shared",
        "type": "b64",
    }
    assert init.predicate is None
    assert str(init.result_type) == ""

    wait = decode_ptx_call(_call(func, "tirx.ptx.tcgen05_wait"))
    assert dict(wait.operands) == {}
    assert dict(wait.modifiers) == {
        "action": "wait::ld",
        "sync": "sync",
        "aligned": "aligned",
    }

    cuda_sync = _call(func, "tirx.cuda.cta_sync")
    with pytest.raises(PtxCallDecodeError, match=r"only applies to tirx\.ptx\.\*"):
        decode_ptx_call(cuda_sync)


@pytest.mark.parametrize(
    ("source", "message"),
    [
        (
            """
@T.prim_func
def invalid():
    T.device_entry()
    T.ptx.tcgen05.shift(T.uint32(0), cta_group=1)
""",
            r"'shift' is not a valid modifier for 'tcgen05'",
        ),
        (
            """
@T.prim_func
def invalid():
    T.device_entry()
    T.ptx["tcgen05.mma.cta_group::1.kind::tf32"](
        T.uint32(0),
        T.uint32(0),
        T.uint64(0),
        T.uint32(0),
        T.uint32(0),
        T.uint32(0),
        T.uint32(0),
        T.uint32(0),
        T.ptx.pred(T.uint32(0)),
        T.uint32(1),
    )
""",
            r"expects 9 operand\(s\).*got 10",
        ),
        (
            """
@T.prim_func
def invalid():
    T.device_entry()
    T.ptx.mma.legacy()
""",
            r"'legacy' is not a valid modifier for 'mma'",
        ),
    ],
)
def test_target_parser_rejects_removed_tcgen05_families(source: str, message: str):
    with pytest.raises(tvm.error.DiagnosticError, match=message):
        _parse(source)


def test_target_parser_rejects_tensormap_prefetch_in_global_space():
    with pytest.raises(
        tvm.error.DiagnosticError,
        match=r"\.tensormap takes \.const or \.param",
    ):
        _parse(
            """
@T.prim_func
def invalid(tensor_map: T.TensorMap()):
    T.device_entry()
    T.evaluate(T.ptx.prefetch.global_.tensormap(T.address_of(tensor_map)))
"""
        )


def test_decode_dynamic_operand_groups_by_slot_name():
    func = _parse(
        """
@T.prim_func
def kernel(src: T.handle):
    source = T.match_buffer(src, (4,), "uint32")
    T.device_entry()
    r0 = T.local_scalar("uint32")
    r1 = T.local_scalar("uint32")
    r2 = T.local_scalar("uint32")
    r3 = T.local_scalar("uint32")
    T.ptx.ld.global_.v4.u32(r0, r1, r2, r3, source.ptr_to([0]))
"""
    )

    decoded = decode_ptx_call(_call(func, "tirx.ptx.ld_vec"))
    assert len(decoded.operand("d")) == 4
    assert len(decoded.operand("addr")) == 1
    assert decoded.operand("cache_policy") == ()
    assert decoded.modifier("space") == "global"
    assert decoded.modifier("vec") == "v4"
    assert decoded.modifier("type") == "u32"


def test_decode_reconstructs_sink_lanes_without_exposing_marker_positions():
    func = _parse(
        """
@T.prim_func
def kernel():
    T.device_entry()
    hi = T.local_scalar("uint32")
    packed = T.local_scalar("uint64")
    T.ptx.mov.b64(T.ptx.SINK, hi, packed)
"""
    )

    ptx_calls = [call for call in _calls(func, "tirx.ptx.")]
    assert len(ptx_calls) == 1
    decoded = decode_ptx_call(ptx_calls[0])
    destination = decoded.operand("d")
    assert destination[0] is PTX_SINK
    assert destination[1] is not PTX_SINK
    assert len(decoded.operand("a")) == 1

    reparsed = tvm.script.from_source(func.script(), {"T": T})
    tvm.ir.assert_structural_equal(func, reparsed)
    reparsed_call = [call for call in _calls(reparsed, "tirx.ptx.")][0]
    assert decode_ptx_call(reparsed_call).operand("d")[0] is PTX_SINK


def test_decoded_call_and_sink_survive_frontend_cache_pickling():
    func = _parse(
        """
@T.prim_func
def kernel():
    T.device_entry()
    hi = T.local_scalar("uint32")
    packed = T.local_scalar("uint64")
    T.ptx.mov.b64(T.ptx.SINK, hi, packed)
"""
    )

    decoded = decode_ptx_call(_calls(func, "tirx.ptx.")[0])
    restored = pickle.loads(pickle.dumps(decoded))

    assert isinstance(restored, DecodedPtxCall)
    assert restored.op_name == decoded.op_name
    assert dict(restored.modifiers) == dict(decoded.modifiers)
    assert restored.operand("d")[0] is PTX_SINK


def test_decode_materializes_table_owned_literal_operands():
    func = _parse(
        """
@T.prim_func
def kernel():
    T.device_entry()
    shared = T.alloc_buffer((8,), "uint8", scope="shared")
    T.ptx.st_bulk.shared__cta(shared.ptr_to([0]), T.uint64(8))
"""
    )

    decoded = decode_ptx_call(_call(func, "tirx.ptx.st_bulk"))

    assert tuple(decoded.operands) == ("addr", "size", "initval")
    assert decoded.operand("initval") == ("0",)


def test_emitter_decodes_only_table_driven_ptx_calls():
    func = _parse(
        """
@T.prim_func
def kernel():
    T.device_entry()
    bar = T.alloc_shared((1,), "uint64")
    T.ptx.mbarrier.init.shared.b64(bar.ptr_to([0]), T.uint32(1))
    T.cuda.cta_sync()
    a = T.alloc_local((8,), "float16")
    b = T.alloc_local((4,), "float16")
    c = T.alloc_local((4,), "float32")
    T.ptx_legacy.mma(
        "m16n8k16", "row", "col", "float16", "float16", "float32",
        a.data, 0, b.data, 0, c.data, 0, False, dtype="float32"
    )
"""
    )

    kernel = resolved_kernel(func)
    assert {"tirx.ptx.mbarrier_init", "tirx.cuda.cta_sync", "tirx.ptx_legacy.mma"} <= (
        call_op_names(kernel)
    )

    ptx_call = _call(func, "tirx.ptx.mbarrier_init")
    decoded = decode_ptx_call(ptx_call)
    assert isinstance(decoded, DecodedPtxCall)
    assert decoded.op_name == "tirx.ptx.mbarrier_init"

    cuda_call = _call(func, "tirx.cuda.cta_sync")
    with pytest.raises(PtxCallDecodeError, match=r"only applies to tirx\.ptx\.\*"):
        decode_ptx_call(cuda_call)

    legacy_call = _call(func, "tirx.ptx_legacy.mma")
    with pytest.raises(PtxCallDecodeError, match=r"only applies to tirx\.ptx\.\*"):
        decode_ptx_call(legacy_call)


def test_decode_reconstructs_register_class_and_instruction_predicate():
    func = _parse(
        """
@T.prim_func
def kernel(dst: T.handle):
    out = T.match_buffer(dst, (1,), "uint32")
    T.device_entry()
    tmem = T.local_scalar("uint32")
    desc = T.local_scalar("uint64")
    idesc = T.local_scalar("uint32")
    flag = T.local_scalar("uint32")
    T.ptx["tcgen05.mma.cta_group::1.kind::f16"](
        tmem, desc, desc, idesc, 0, 0, 0, 0, T.ptx.pred(flag)
    )
    T.ptx.red.relaxed.gpu.global_.add.u32(out.ptr_to([0]), T.uint32(1), pred=flag)
"""
    )

    mma_call = [call for call in _calls(func, "tirx.ptx.tcgen05_mma_")][0]
    mma = decode_ptx_call(mma_call)
    assert len(mma.operand("enable_input_d")) == 1
    assert mma.predicate is None

    red = decode_ptx_call(_call(func, "tirx.ptx.red"))
    assert tuple(red.operands) == ("addr", "value", "cache_policy")
    assert red.operand("cache_policy") == ()
    assert red.predicate is not None
    assert red.modifier("sem") == "relaxed"
    assert red.modifier("scope") == "gpu"


@pytest.mark.parametrize(
    "mutate",
    [
        lambda call: (*call.args[:-1], ir.StringImm("s99")),
        lambda call: (*call.args[:-1], ir.StringImm("p0,p0")),
        lambda call: (*call.args[:-4], ir.StringImm("bogus"), *call.args[-3:]),
        lambda call: call.args[:-1],
    ],
)
def test_decode_rejects_corrupt_serialized_metadata(mutate):
    func = _parse(
        """
@T.prim_func
def kernel(bar: T.handle):
    T.device_entry()
    T.ptx.mbarrier.init.shared.b64(bar, T.uint32(1))
"""
    )
    call = _call(func, "tirx.ptx.mbarrier_init")
    corrupted = _with_args(call, mutate(call))

    with pytest.raises(PtxCallDecodeError) as caught:
        decode_ptx_call(corrupted)
    assert "tirx.ptx.mbarrier_init" in str(caught.value)
    assert caught.value.source_span.same_as(call.span)
