from __future__ import annotations

import itertools
from typing import NamedTuple

import pytest
import tvm
from tvm import ir, tirx
from tvm.ir import Expr
from tvm.ir.type import PointerType
from tvm.tirx import TensorMapType
from tvm.script import tirx as T
from tvm.tirx import Stmt, Var
from tvm_ffi import structural_walk

from tirx_harness.numsim.errors import UnsupportedTIRxError
from tirx_harness.numsim.host_abi import HostAbiError, build_host_abi
from tests.numsim.support.kernels import (
    no_op_kernel,
    raw_tma_roundtrip,
    tcgen_lifecycle_single_cta,
    tcgen_tmem_to_local_roundtrip,
)
from tests.numsim.support.manifest import (
    call_op_names,
    device_kernel,
    emitted_module,
    emitted_calls,
    evaluated_kernel,
    replace_call,
    resolved_kernel,
)
from tirx_harness.numsim.transpiler.frontend import TensorMapSpec, analyze

_HANDLE = PointerType(ir.PrimType(""))
_ARG_NAMES = itertools.count()


class _Form(NamedTuple):
    func: tirx.PrimFunc
    op_name: str


def _result_type(dtype: str) -> ir.Type:
    return _HANDLE if dtype == "handle" else ir.PrimType(dtype)


def _arg(dtype: str) -> Var:
    return Var(f"arg_{next(_ARG_NAMES)}", dtype)


def _call(op_name: str, result_dtype: str, *args: object) -> ir.Call:
    return ir.Call(ir.Op.get(op_name), list(args), ret_ty=_result_type(result_dtype))


def _form(call: ir.Call) -> _Form:
    params = [argument for argument in call.args if isinstance(argument, Var)]
    return _Form(evaluated_kernel(call, params), str(call.op.name))


def _emitted_heads(form: _Form) -> list[str]:
    """The engine instructions the form's call site emits, in body order."""

    return [call.head for call in emitted_calls(*form)]


def _matrix_descriptor_call(fields: str) -> _Form:
    return _public_ptx_call(
        f"""
@T.prim_func
def kernel(ldo: T.int32, sdo: T.int32, swizzle: T.int32):
    T.device_entry()
    shared = T.alloc_buffer((64,), "uint32", scope="shared")
    descriptor: T.uint64
    T.cuda.tcgen05.encode_matrix_descriptor(
        T.address_of(descriptor), T.address_of(shared[4]), {fields}
    )
""",
        "tirx.cuda.tcgen05_encode_matrix_descriptor",
    )


def test_matrix_descriptor_fields_are_runtime_operands_not_specializations():
    instructions = {
        tuple(_emitted_heads(_matrix_descriptor_call(fields)))
        for fields in ("1, 8, 3", "-1, 0x4000, -1", "ldo, sdo, swizzle")
    }
    assert instructions == {
        (
            "v2::addr::cvta::<v2::addr::variant::GenericToSharedU32<false>>",
            "v2::mem::st::<v2::mem::variant::St<v2::reg::variant::U64, v2::Generic>>",
        )
    }


def _public_ptx_func_and_call(source: str, op_name: str) -> tuple[object, object]:
    func = tvm.script.from_source(source, {"T": T})
    calls = []

    def visit(node: object) -> None:
        if type(node).__name__ == "Call" and str(getattr(node.op, "name", "")) == op_name:
            calls.append(node)

    structural_walk(func.body, ((Expr, Stmt), visit))
    assert len(calls) == 1
    return func, calls[0]


def _public_ptx_call(source: str, op_name: str) -> _Form:
    return _Form(_public_ptx_func_and_call(source, op_name)[0], op_name)


def _public_tensormap_prefetch(parameter_name: str = "tensor_map") -> tuple[object, object]:
    return _public_ptx_func_and_call(
        f"""
@T.prim_func
def kernel({parameter_name}: T.TensorMap()):
    T.device_entry()
    T.evaluate(T.ptx.prefetch.tensormap(T.address_of({parameter_name})))
""",
        "tirx.ptx.prefetch",
    )


def _cta_reduce_call(operation: str, num_warps: int, *, dtype: str = "float32") -> _Form:
    # The scratch operand must be a shared address the emitter can bind, so the form
    # replaces the call of a public `cta_sum` kernel instead of taking a bare handle.
    func, public = _public_ptx_func_and_call(
        """
@T.prim_func
def kernel():
    T.device_entry()
    scratch = T.alloc_buffer((1,), "float32", scope="shared")
    T.evaluate(T.cuda.cta_sum(T.float32(0), 1, scratch.ptr_to([0])))
""",
        "tirx.cuda.cta_reduce",
    )
    value = _arg(dtype)
    call = _call(
        "tirx.cuda.cta_reduce",
        dtype,
        value,
        ir.StringImm(operation),
        tirx.IntImm("int32", num_warps),
        public.args[3],
    )
    body = replace_call(func, public, call).body
    return _Form(tirx.PrimFunc([value], body, func.ret_type, func.attrs), "tirx.cuda.cta_reduce")


def _warp_reduce_call(operation: str, width: int, *, dtype: str = "float32") -> _Form:
    return _form(
        _call(
            "tirx.cuda.warp_reduce",
            dtype,
            _arg(dtype),
            ir.StringImm(operation),
            tirx.IntImm("int32", width),
        )
    )


def _fetch_register_call(bits: int, register: str) -> _Form:
    return _form(
        _call(
            "tirx.cuda.mov_sreg",
            f"int{bits}",
            tirx.IntImm("int32", bits),
            ir.StringImm(register),
        )
    )


def _mapa_call(*, space: str = "", ptx_type: str = "u64") -> _Form:
    if ptx_type == "u32":
        if space != "shared::cluster":
            raise ValueError("mapa.u32 exists only in shared::cluster space")
        return _public_ptx_call(
            """
@T.prim_func
def kernel():
    T.device_entry()
    address = T.local_scalar("uint32")
    mapped = T.local_scalar("uint32")
    T.ptx.mapa.shared__cluster.u32(mapped, address, T.uint32(0))
""",
            "tirx.ptx.mapa_u32",
        )
    if ptx_type != "u64" or space not in {"", "shared::cluster"}:
        raise ValueError(f"unsupported public mapa form {space=}, {ptx_type=}")
    space_chain = "shared__cluster." if space else ""
    return _public_ptx_call(
        f"""
@T.prim_func
def kernel():
    T.device_entry()
    shared = T.alloc_buffer((1,), "uint64", scope="shared")
    mapped = T.local_scalar("uint64")
    T.ptx.mapa.{space_chain}u64(mapped, T.address_of(shared[0]), T.uint32(0))
""",
        "tirx.ptx.mapa",
    )


def _mbarrier_parity_call(*, sem: str = "", scope: str = "", space: str = "shared::cta") -> _Form:
    sem_scope = f".{sem}.{scope}" if sem else ""
    space_token = {"shared": "shared", "shared::cta": "shared__cta"}[space]
    return _public_ptx_call(
        f"""
@T.prim_func
def kernel():
    T.device_entry()
    bar = T.alloc_buffer((1,), "uint64", scope="shared")
    ready = T.local_scalar("uint32")
    T.ptx.mbarrier.test_wait.parity{sem_scope}.{space_token}.b64(
        ready, T.address_of(bar[0]), T.uint32(0)
    )
""",
        "tirx.ptx.mbarrier_test_wait_parity",
    )


def _mbarrier_try_wait_call(time_hint: object, *, sem: str = "acquire") -> _Form:
    if type(time_hint).__name__ == "IntImm":
        parameter = ""
        argument = f"T.uint32({int(time_hint.value)})"
    else:
        parameter = "time_hint: T.uint32"
        argument = "time_hint"
    sem_scope = f".{sem}.cta" if sem else ""
    return _public_ptx_call(
        f"""
@T.prim_func
def kernel({parameter}):
    T.device_entry()
    bar = T.alloc_buffer((1,), "uint64", scope="shared")
    ready = T.local_scalar("uint32")
    T.ptx.mbarrier.try_wait.parity{sem_scope}.shared__cta.b64(
        ready, T.address_of(bar[0]), T.uint32(0), {argument}
    )
""",
        "tirx.ptx.mbarrier_try_wait_parity",
    )


def _mbarrier_try_wait_no_hint_call(*, sem: str = "") -> _Form:
    sem_scope = f".{sem}.cta" if sem else ""
    return _public_ptx_call(
        f"""
@T.prim_func
def kernel():
    T.device_entry()
    bar = T.alloc_buffer((1,), "uint64", scope="shared")
    ready = T.local_scalar("uint32")
    T.ptx.mbarrier.try_wait.parity{sem_scope}.shared__cta.b64(
        ready, T.address_of(bar[0]), T.uint32(0)
    )
""",
        "tirx.ptx.mbarrier_try_wait_parity_no_hint",
    )


def _cluster_wait_call(*, acquire: bool, aligned: bool) -> _Form:
    modifiers = (".acquire" if acquire else "") + (".aligned" if aligned else "")
    return _public_ptx_call(
        f"""
@T.prim_func
def kernel():
    T.device_entry()
    T.ptx.barrier.cluster.wait{modifiers}()
""",
        "tirx.ptx.barrier_cluster_wait",
    )


def test_resolver_keeps_the_validated_tirx_call_as_the_lowering_input():
    call = _call("tirx.fma", "float32", _arg("float32"), _arg("float32"), _arg("float32"))

    assert _emitted_heads(_form(call)) == ["v2::reg::fma::<v2::reg::variant::F32Rn>"]


@pytest.mark.parametrize(
    ("left", "right"),
    [
        (_cta_reduce_call("sum", 4), _cta_reduce_call("max", 4)),
        (_warp_reduce_call("sum", 32), _warp_reduce_call("min", 32)),
    ],
)
def test_pure_call_static_form_mutations_change_the_emitted_instruction(left: _Form, right: _Form):
    assert _emitted_heads(left) != _emitted_heads(right)


def test_fetch_register_alias_normalization_is_explicit():
    plain = _fetch_register_call(32, "laneid")
    percent_prefixed = _fetch_register_call(32, "%laneid")
    # `mov_sreg` lowers inline, so the whole body is the alias witness.
    assert emitted_module(plain.func) == emitted_module(percent_prefixed.func)


def test_removed_map_shared_rank_alias_is_not_public():
    assert not hasattr(T.ptx, "map_shared_rank")


_REDUCTION_DTYPES = {
    "int8",
    "int16",
    "int32",
    "int64",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
    "float16",
    "bfloat16",
    "float32",
    "float64",
}

_CTA_REDUCE_VALUES = {
    "int8": "v2::reg::variant::I8",
    "int16": "v2::reg::variant::I16",
    "int32": "v2::reg::variant::I32",
    "int64": "v2::reg::variant::I64",
    "uint8": "v2::reg::variant::U8",
    "uint16": "v2::reg::variant::U16",
    "uint32": "v2::reg::variant::U32",
    "uint64": "v2::reg::variant::U64",
    "float16": "v2::collective::variant::F16",
    "bfloat16": "v2::collective::variant::Bf16",
    "float32": "v2::reg::variant::F32",
    "float64": "v2::reg::variant::F64",
}


def test_reduction_static_form_domains_are_closed_and_exhaustive():
    operations = {"sum", "max", "min"}
    warp_widths = {2, 4, 8, 16, 32}
    cta_warp_counts = {1, 2, 4, 8, 16, 32}

    for dtype in _REDUCTION_DTYPES:
        for operation in operations:
            cta_instruction = [
                "v2::collective::cta_reduce::<v2::collective::variant::Reduce<"
                f"{_CTA_REDUCE_VALUES[dtype]}, v2::collective::variant::{operation.title()}>>"
            ]
            for count in cta_warp_counts:
                assert (
                    _emitted_heads(_cta_reduce_call(operation, count, dtype=dtype))
                    == cta_instruction
                )
            for width in warp_widths:
                # A warp reduction lowers to butterfly shuffles plus a combiner per step;
                # the width itself stays an operand of the shuffles.
                instructions = _emitted_heads(_warp_reduce_call(operation, width, dtype=dtype))
                assert instructions[0].startswith("v2::warp::shfl_sync::<v2::warp::variant::Shfl<")

    for invalid in (1, 3, 6, 33):
        with pytest.raises(UnsupportedTIRxError, match="power of two"):
            resolved_kernel(_warp_reduce_call("sum", invalid).func)
    for invalid in (0, 3, 6, 64):
        with pytest.raises(UnsupportedTIRxError, match="power of two"):
            resolved_kernel(_cta_reduce_call("sum", invalid).func)
    for operation in ("add", "maximum", ""):
        with pytest.raises(UnsupportedTIRxError, match="operation"):
            resolved_kernel(_warp_reduce_call(operation, 32).func)
        with pytest.raises(UnsupportedTIRxError, match="operation"):
            resolved_kernel(_cta_reduce_call(operation, 4).func)
    for dtype in ("bool", "float8_e4m3", "float16x2", "handle"):
        with pytest.raises(UnsupportedTIRxError, match="supported numeric scalar dtype"):
            resolved_kernel(_warp_reduce_call("sum", 32, dtype=dtype).func)
        with pytest.raises(UnsupportedTIRxError, match="supported numeric scalar dtype"):
            resolved_kernel(_cta_reduce_call("sum", 4, dtype=dtype).func)


def test_fetch_register_form_domain_is_the_exact_modeled_register_set():
    registers32 = {
        "tid.x",
        "tid.y",
        "tid.z",
        "ntid.x",
        "ntid.y",
        "ntid.z",
        "laneid",
        "warpid",
        "nwarpid",
        "smid",
        "ctaid.x",
        "ctaid.y",
        "ctaid.z",
        "nctaid.x",
        "nctaid.y",
        "nctaid.z",
        "clusterid.x",
        "clusterid.y",
        "clusterid.z",
        "nclusterid.x",
        "nclusterid.y",
        "nclusterid.z",
        "cluster_ctaid.x",
        "cluster_ctaid.y",
        "cluster_ctaid.z",
        "cluster_nctaid.x",
        "cluster_nctaid.y",
        "cluster_nctaid.z",
        "cluster_ctarank",
        "cluster_nctarank",
        "lanemask_eq",
        "lanemask_le",
        "lanemask_lt",
        "lanemask_ge",
        "lanemask_gt",
        "clock",
        "clock_hi",
        "globaltimer_lo",
        "globaltimer_hi",
    }
    registers64 = {"gridid", "clock64", "globaltimer"}

    # `mov_sreg` lowers inline, so each modeled register is observable only as acceptance.
    for bits, registers in ((32, registers32), (64, registers64)):
        for register in registers:
            resolved_kernel(_fetch_register_call(bits, register).func)

    for bits, register in (
        (32, "pm0"),
        (32, "envreg0"),
        (32, "gridid"),
        (64, "laneid"),
    ):
        with pytest.raises(UnsupportedTIRxError, match="is not modeled"):
            resolved_kernel(_fetch_register_call(bits, register).func)
    for bits in (16, 128):
        with pytest.raises(UnsupportedTIRxError, match="static 32 or 64"):
            resolved_kernel(
                _form(
                    _call(
                        "tirx.cuda.mov_sreg",
                        "int32",
                        tirx.IntImm("int32", bits),
                        ir.StringImm("laneid"),
                    )
                ).func
            )


def test_mapa_and_mbarrier_static_form_domains_are_explicit():
    # `mapa` lowers inline, so each modeled form is observable only as acceptance.
    for space, ptx_type in (
        ("", "u64"),
        ("shared::cluster", "u64"),
        ("shared::cluster", "u32"),
    ):
        resolved_kernel(_mapa_call(space=space, ptx_type=ptx_type).func)

    modifier_pairs = {
        ("", ""),
        ("acquire", "cta"),
        ("acquire", "cluster"),
        ("relaxed", "cta"),
        ("relaxed", "cluster"),
    }
    # Ten accepted modifier spellings, and only `relaxed` selects its own instruction.
    for sem, scope in modifier_pairs:
        relaxed = "Relaxed" if sem == "relaxed" else ""
        for space in ("shared", "shared::cta"):
            assert _emitted_heads(_mbarrier_parity_call(sem=sem, scope=scope, space=space)) == [
                f"v2::sync::mbarrier_test_wait::<v2::sync::variant::TestWaitParity{relaxed}>"
            ]

    static_try_forms = {
        tuple(_emitted_heads(_mbarrier_try_wait_call(tirx.IntImm("uint32", ticks))))
        for ticks in (0, 1, 0xFFFFFFFF)
    }
    dynamic_try_instructions = tuple(
        _emitted_heads(_mbarrier_try_wait_call(tirx.Var("ticks", "uint32")))
    )
    assert len(static_try_forms) == 1
    assert dynamic_try_instructions in static_try_forms
    assert dynamic_try_instructions == (
        "v2::sync::mbarrier_try_wait::<v2::sync::variant::TryWaitParity>",
    )
    assert _emitted_heads(_mbarrier_try_wait_call(tirx.IntImm("uint32", 0), sem="relaxed")) == [
        "v2::sync::mbarrier_try_wait::<v2::sync::variant::TryWaitParityRelaxed>"
    ]


def test_mbarrier_ticks_are_runtime_operands_not_instruction_variants():
    static = _mbarrier_try_wait_call(tirx.IntImm("uint32", 17))
    dynamic = _mbarrier_try_wait_call(tirx.Var("ticks", "uint32"))

    assert _emitted_heads(static) == _emitted_heads(dynamic)


def test_mbarrier_try_wait_without_hint_uses_the_same_wait_parity_specialization():
    assert _emitted_heads(_mbarrier_try_wait_no_hint_call()) == _emitted_heads(
        _mbarrier_try_wait_call(tirx.IntImm("uint32", 1), sem="")
    )


def _fence_call(sem: str, scope: str) -> _Form:
    return _public_ptx_call(
        f"""
@T.prim_func
def kernel():
    T.device_entry()
    T.ptx.fence.{sem}.{scope}()
""",
        "tirx.ptx.fence",
    )


def test_fence_specializes_on_scope_only_and_still_validates_order():
    instructions = {
        _emitted_heads(_fence_call(sem, scope))[0]
        for sem in ("sc", "acq_rel")
        for scope in ("cta", "cluster", "gpu", "sys")
    }

    # SC remains distinct metadata; its extra ordering is a deferred model gap.
    assert len(instructions) == 8
    for scope_marker in ("Cta", "Cluster", "Gpu", "Sys"):
        assert any(
            f"Fence<v2::sync::variant::{scope_marker}>" in instruction
            for instruction in instructions
        )
    for scope in ("cta", "cluster", "gpu", "sys"):
        assert _emitted_heads(_fence_call("sc", scope)) != _emitted_heads(
            _fence_call("acq_rel", scope)
        )

    assert _emitted_heads(_fence_call("release", "gpu")) == [
        "v2::sync::fence::<v2::sync::variant::Fence<v2::sync::variant::Gpu, "
        "v2::sync::variant::Release>>"
    ]
    with pytest.raises(tvm.error.DiagnosticError):
        _fence_call("sc", "block")


def test_cluster_arrive_transcribes_each_semantics_spelling_verbatim():
    def arrive_call(sem: str) -> _Form:
        modifier = f".{sem}" if sem else ""
        return _public_ptx_call(
            f"""
@T.prim_func
def kernel():
    T.device_entry()
    T.ptx.barrier.cluster.arrive{modifier}.aligned()
""",
            "tirx.ptx.barrier_cluster_arrive",
        )

    # One marker per source spelling — the unqualified default keeps its own
    # marker instead of folding into the explicit release form.
    by_sem = {sem: _emitted_heads(arrive_call(sem))[0] for sem in ("", "release", "relaxed")}
    assert "ClusterArrive<v2::sync::variant::DefaultRelease, true>" in by_sem[""]
    assert "ClusterArrive<v2::sync::variant::Release, true>" in by_sem["release"]
    assert "ClusterArrive<v2::sync::variant::Relaxed, true>" in by_sem["relaxed"]


def test_cluster_wait_specializes_on_aligned_only_and_still_validates_acquire():
    instructions = {
        _emitted_heads(_cluster_wait_call(acquire=acquire, aligned=aligned))[0]
        for acquire in (False, True)
        for aligned in (False, True)
    }

    # `barrier.cluster.wait` has no wait-side semantics axis: PTX defines the
    # unqualified form as acquire and the engine models exactly one acquiring
    # wait, so the acquire qualifier selects nothing.
    assert len(instructions) == 2
    assert all("v2::sync::barrier_cluster_wait" in instruction for instruction in instructions)
    assert not any("Acquire" in instruction for instruction in instructions)
    assert any("ClusterWait<true>" in instruction for instruction in instructions)
    assert any("ClusterWait<false>" in instruction for instruction in instructions)

    for aligned in (False, True):
        assert _emitted_heads(_cluster_wait_call(acquire=False, aligned=aligned)) == _emitted_heads(
            _cluster_wait_call(acquire=True, aligned=aligned)
        )

    # The public builder exposes modifiers, not the removed boolean ABI.
    with pytest.raises(tvm.error.DiagnosticError):
        _public_ptx_call(
            """
@T.prim_func
def kernel():
    T.device_entry()
    T.ptx.barrier.cluster.wait(T.bool(True))
""",
            "tirx.ptx.barrier_cluster_wait",
        )


@pytest.mark.parametrize(
    "form",
    [
        _form(
            _call(
                "tirx.cuda.cta_reduce",
                "float32",
                _arg("float32"),
                _arg("handle"),
                tirx.IntImm("int32", 4),
                _arg("handle"),
            )
        ),
        _form(
            _call(
                "tirx.cuda.warp_reduce",
                "float32",
                _arg("float32"),
                _arg("handle"),
                tirx.IntImm("int32", 32),
            )
        ),
        _form(
            _call(
                "tirx.cuda.mov_sreg",
                "int32",
                tirx.IntImm("int32", 32),
                _arg("handle"),
            )
        ),
    ],
)
def test_pure_call_form_strings_must_be_compile_time_static(form: _Form):
    with pytest.raises(UnsupportedTIRxError, match="must be static"):
        resolved_kernel(form.func)


@pytest.mark.parametrize("op_name", ["tirx.break_loop", "tirx.continue_loop"])
def test_resolver_validates_control_calls(op_name: str):
    def looped(call: ir.Call) -> tirx.PrimFunc:
        index = Var("index", "int32")
        loop = tirx.For(
            index,
            tirx.IntImm("int32", 0),
            tirx.IntImm("int32", 4),
            tirx.ForKind.SERIAL,
            tirx.Evaluate(call),
        )
        return device_kernel(loop)

    kernel = resolved_kernel(looped(_call(op_name, "")))
    assert op_name in call_op_names(kernel)

    with pytest.raises(UnsupportedTIRxError, match="expects no arguments"):
        resolved_kernel(looped(_call(op_name, "", _arg("int32"))))


def test_public_mbarrier_init_rejects_non_pointer_operands_during_parsing():
    with pytest.raises(tvm.error.DiagnosticError, match="must be a shared-scope pointer"):
        tvm.script.from_source(
            """
@T.prim_func
def kernel():
    T.device_entry()
    T.ptx.mbarrier.init.shared.b64(T.float32(1), T.float32(1))
""",
            {"T": T},
        )


def _mbarrier_init_call(*, layout: str = "") -> _Form:
    layout_modifier = f".layout__{layout}" if layout else ""
    return _public_ptx_call(
        f"""
@T.prim_func
def kernel():
    T.device_entry()
    bar = T.alloc_buffer((1,), "uint64", scope="shared")
    T.ptx.mbarrier.init{layout_modifier}.shared.b64(
        T.address_of(bar[0]), T.uint32(1)
    )
""",
        "tirx.ptx.mbarrier_init",
    )


def test_mbarrier_init_default_and_layout_v0_resolve_identically():
    assert _emitted_heads(_mbarrier_init_call()) == _emitted_heads(_mbarrier_init_call(layout="v0"))


def test_mbarrier_init_layout_v1_has_distinct_specialization():
    v0 = _emitted_heads(_mbarrier_init_call(layout="v0"))
    v1 = _emitted_heads(_mbarrier_init_call(layout="v1"))
    assert v1 != v0
    assert v1 == ["v2::sync::mbarrier_init::<v2::sync::variant::MbarrierInit<true>>"]
    assert v0 == ["v2::sync::mbarrier_init::<v2::sync::variant::MbarrierInit>"]


def test_atomic_bulk_classifier_rejects_nonvoid_st_bulk():
    func, valid = _public_ptx_func_and_call(
        """
@T.prim_func
def kernel():
    T.device_entry()
    shared = T.alloc_buffer((8,), "uint8", scope="shared")
    T.ptx.st_bulk.shared__cta(shared.ptr_to([0]), T.uint64(8))
""",
        "tirx.ptx.st_bulk",
    )
    nonvoid = ir.Call(
        valid.op,
        list(valid.args),
        attrs=valid.attrs,
        span=valid.span,
        ret_ty=ir.PrimType("float32"),
    )
    with pytest.raises(UnsupportedTIRxError, match="st_bulk must return void"):
        resolved_kernel(replace_call(func, valid, nonvoid))


def test_resolver_rejects_opaque_cuda_helpers_at_the_single_boundary():
    function_name = ir.StringImm("arbitrary_helper")

    with pytest.raises(UnsupportedTIRxError, match="opaque CUDA helper bodies"):
        resolved_kernel(_form(_call("tirx.cuda.func_call", "float32", function_name)).func)


def test_resolver_remains_closed_for_unknown_calls():
    unknown = _call("tirx.tvm_stack_alloca", "handle", ir.StringImm("x"), tirx.IntImm("int32", 1))
    with pytest.raises(UnsupportedTIRxError, match="unregistered raw call"):
        resolved_kernel(_form(unknown).func)


def test_ordinary_handle_address_is_not_classified_as_a_tensor_map():
    ordinary_handle = Var("descriptor_like_name", "handle")
    address = _call("tirx.address_of", "handle", ordinary_handle)

    with pytest.raises(UnsupportedTIRxError, match="address_of requires a concrete TensorLoad"):
        resolved_kernel(evaluated_kernel(address, (ordinary_handle,)))


def test_ptx_tensormap_prefetch_requires_a_typed_owning_parameter():
    func, _prefetch = _public_tensormap_prefetch()

    assert _emitted_heads(_Form(func, "tirx.ptx.prefetch")) == [
        "v2::async_copy::prefetch_tensormap"
    ]

    ordinary_handle = Var("descriptor_like_name", "handle")
    with pytest.raises(UnsupportedTIRxError, match="typed PrimFunc parameter identity"):
        resolved_kernel(tirx.PrimFunc([ordinary_handle], func.body, func.ret_type, func.attrs))


def test_ptx_tensormap_prefetch_rejects_a_nonvoid_result():
    func, prefetch = _public_tensormap_prefetch()
    nonvoid = ir.Call(
        prefetch.op,
        list(prefetch.args),
        attrs=prefetch.attrs,
        span=prefetch.span,
        ret_ty=ir.PrimType("float32"),
    )

    with pytest.raises(UnsupportedTIRxError, match="must return void"):
        resolved_kernel(replace_call(func, prefetch, nonvoid))


def test_tensor_map_resolution_requires_exact_primfunc_parameter_identity():
    func, _prefetch = _public_tensormap_prefetch("same_name")
    kernel = resolved_kernel(func)
    assert kernel.tensor_maps == (TensorMapSpec("same_name", 0),)

    impostor = Var("same_name", PointerType(TensorMapType()))
    with pytest.raises(UnsupportedTIRxError, match="typed PrimFunc parameter identity"):
        resolved_kernel(tirx.PrimFunc([impostor], func.body, func.ret_type, func.attrs))


def test_frontend_discovers_typed_tensor_map_parameters_from_exact_owners():
    spec = analyze(raw_tma_roundtrip).kernels[0]

    assert spec.tensor_maps == (
        TensorMapSpec("input_map", 0),
        TensorMapSpec("output_map", 1),
    )
    assert "tirx.address_of" in call_op_names(spec)


def test_host_abi_rejects_distinct_tensor_map_parameters_with_one_public_name():
    first_func, first_prefetch = _public_tensormap_prefetch("descriptor")
    second_func, second_prefetch = _public_tensormap_prefetch("descriptor")
    first = first_func.params[0]
    second = second_func.params[0]
    body = tirx.SeqStmt(
        [
            tirx.Evaluate(first_prefetch),
            tirx.Evaluate(second_prefetch),
        ]
    )
    spec = analyze(tirx.PrimFunc([first, second], body))
    assert spec.kernels[0].tensor_maps == (
        TensorMapSpec("descriptor", 0),
        TensorMapSpec("descriptor", 1),
    )

    with pytest.raises(HostAbiError, match='host binding "descriptor"'):
        build_host_abi(spec)


def test_frontend_records_typed_implicit_tmem_requirement():
    assert analyze(tcgen_lifecycle_single_cta).kernels[0].requires_implicit_tmem
    assert not analyze(tcgen_tmem_to_local_roundtrip).kernels[0].requires_implicit_tmem
    assert not analyze(no_op_kernel).kernels[0].requires_implicit_tmem


def _cvt_call(
    spelling: str,
    *,
    result_dtype: str,
    source_dtype: str,
) -> _Form:
    """Build one scalar CVT through the target TVM's public exact-op surface."""

    func = tvm.script.from_source(
        f"""
@T.prim_func
def kernel(
    destination: T.Buffer((1,), "{result_dtype}"),
    source: T.Buffer((1,), "{source_dtype}"),
):
    T.device_entry()
    T.ptx["{spelling}"](destination[0], source[0])
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
    return _Form(func, str(calls[0].op.name))


_VARIANT = "v2::reg::variant::"
_CVT_TYPES = {
    "s8": "I8",
    "s16": "I16",
    "s32": "I32",
    "s64": "I64",
    "u8": "U8",
    "u16": "U16",
    "u32": "U32",
    "u64": "U64",
    "f16": "F16",
    "bf16": "Bf16",
    "f32": "F32",
    "f64": "F64",
    "tf32": "Tf32",
    "ue5m3x2": "Ue5m3x2",
}


def _cvt_mode(kind: str, *markers: str) -> str:
    """One CVT mode variant, such as ``CvtMode<Rn, PreserveSubnormal>``."""

    return f"{_VARIANT}{kind}<" + ", ".join(_VARIANT + marker for marker in markers) + ">"


def _cvt_variant(source: str, destination: str, mode: str = f"{_VARIANT}Unmodified") -> str:
    """The generics of the engine CVT instruction one scalar spelling selects."""

    return (
        f"{_VARIANT}Cvt<{_VARIANT}{_CVT_TYPES[source]}, "
        f"{_VARIANT}{_CVT_TYPES[destination]}, {mode}>"
    )


def _emitted_cvt(form: _Form) -> str:
    """The CVT generics the site emits; a store of the result follows it."""

    conversion = emitted_calls(*form)[0]
    assert conversion.function == "v2::reg::cvt"
    return conversion.generics


def test_ue5m3_down_conversion_preserves_absence_of_satfinite():
    call = _public_ptx_call(
        """
@T.prim_func
def kernel():
    T.device_entry()
    destination = T.local_scalar("uint16")
    first = T.local_scalar("float32")
    second = T.local_scalar("float32")
    T.ptx["cvt.rp.ue5m3x2.f32"](destination, first, second)
""",
        "tirx.ptx.cvt_ue5m3x2_f32",
    )
    assert _emitted_cvt(call) == _cvt_variant(
        "f32", "ue5m3x2", _cvt_mode("PackedMode", "Rp", "NoSatFinite", "NoRelu")
    )


def test_scalar_cvt_wide_carrier_keeps_the_same_instruction_variant():
    wide = _cvt_call("cvt.u32.s32", result_dtype="uint64", source_dtype="int32")
    canonical = _cvt_call("cvt.u32.s32", result_dtype="uint32", source_dtype="int32")
    assert _emitted_cvt(wide) == _emitted_cvt(canonical)


def test_modeled_scalar_cvt_forms_select_the_exact_modifier_variant():
    for dtype in ("f16", "bf16"):
        for rounding in ("rni", "rzi", "rmi", "rpi"):
            assert _emitted_cvt(
                _cvt_call(
                    f"cvt.{rounding}.{dtype}.{dtype}", result_dtype="uint16", source_dtype="uint16"
                )
            ) == _cvt_variant(
                dtype, dtype, _cvt_mode("CvtMode", rounding.title(), "PreserveSubnormal")
            )
    for rounding in ("", "rni", "rzi", "rmi", "rpi"):
        assert _emitted_cvt(
            _cvt_call(
                f"cvt.{rounding + '.' if rounding else ''}sat.f16.f16",
                result_dtype="uint16",
                source_dtype="uint16",
            )
        ) == _cvt_variant(
            "f16",
            "f16",
            _cvt_mode("CvtMode", rounding.title() or "Unmodified", "PreserveSubnormal", "Sat"),
        )
    for source in ("f16", "bf16"):
        assert _emitted_cvt(
            _cvt_call(f"cvt.f64.{source}", result_dtype="float64", source_dtype="uint16")
        ) == _cvt_variant(source, "f64")
    for destination, source in (("f16", "f64"), ("bf16", "f64"), ("bf16", "f16"), ("f16", "bf16")):
        for rounding in ("rn", "rz", "rm", "rp"):
            assert _emitted_cvt(
                _cvt_call(
                    f"cvt.{rounding}.{destination}.{source}",
                    result_dtype="uint16",
                    source_dtype="float64" if source == "f64" else "uint16",
                )
            ) == _cvt_variant(
                source, destination, _cvt_mode("CvtMode", rounding.title(), "PreserveSubnormal")
            )
    assert _emitted_cvt(
        _cvt_call("cvt.rn.sat.f32.f64", result_dtype="float32", source_dtype="float64")
    ) == _cvt_variant("f64", "f32", _cvt_mode("CvtMode", "Rn", "PreserveSubnormal", "Sat"))
    for destination in ("f16", "bf16"):
        for source in ("s16", "s32", "s64", "u16", "u32", "u64"):
            for rounding in ("rn", "rz", "rm", "rp"):
                assert _emitted_cvt(
                    _cvt_call(
                        f"cvt.{rounding}.{destination}.{source}",
                        result_dtype="uint16",
                        source_dtype=("int" if source[0] == "s" else "uint") + source[1:],
                    )
                ) == _cvt_variant(
                    source, destination, _cvt_mode("CvtMode", rounding.title(), "PreserveSubnormal")
                )
    # An 8-bit integer is exact in f16, so the rounding modifier selects the exact mode.
    for source in ("s8", "u8"):
        for rounding in ("rn", "rz", "rm", "rp"):
            assert _emitted_cvt(
                _cvt_call(
                    f"cvt.{rounding}.f16.{source}",
                    result_dtype="uint16",
                    source_dtype=("int" if source[0] == "s" else "uint") + source[1:],
                )
            ) == _cvt_variant(source, "f16", _cvt_mode("CvtMode", "Exact", "PreserveSubnormal"))
    # Saturating integer-to-float conversion always produces exactly 0 or 1;
    # rounding is inert even when the unsaturated conversion would round.
    for destination in ("f16", "f32", "f64"):
        for source in ("s32", "s64", "u64"):
            for rounding in ("rn", "rz", "rm", "rp"):
                assert _emitted_cvt(
                    _cvt_call(
                        f"cvt.{rounding}.sat.{destination}.{source}",
                        result_dtype="uint16"
                        if destination == "f16"
                        else "float" + destination[1:],
                        source_dtype=("int" if source[0] == "s" else "uint") + source[1:],
                    )
                ) == _cvt_variant(
                    source,
                    destination,
                    _cvt_mode("CvtMode", "Exact", "PreserveSubnormal", "Sat"),
                )
    assert _emitted_cvt(
        _cvt_call("cvt.sat.s8.s32", result_dtype="int8", source_dtype="int32")
    ) == _cvt_variant("s32", "s8", _cvt_mode("CvtMode", "Unmodified", "PreserveSubnormal", "Sat"))
    assert _emitted_cvt(
        _cvt_call("cvt.rn.f32.s32", result_dtype="float32", source_dtype="int32")
    ) == _cvt_variant("s32", "f32", _cvt_mode("CvtMode", "Rn", "PreserveSubnormal"))
    # Float-to-integer `.sat` binds to the same engine entry as its plain form, and the
    # emitted instruction no longer separates them.
    float_to_integer = _cvt_variant("f32", "s32", _cvt_mode("CvtMode", "Rzi", "PreserveSubnormal"))
    assert (
        _emitted_cvt(_cvt_call("cvt.rzi.s32.f32", result_dtype="int32", source_dtype="float32"))
        == float_to_integer
    )
    assert (
        _emitted_cvt(
            _cvt_call(
                "cvt.rzi.sat.s32.f32",
                result_dtype="int32",
                source_dtype="float32",
            )
        )
        == float_to_integer
    )
    assert _emitted_cvt(
        _cvt_call(
            "cvt.rna.satfinite.tf32.f32",
            result_dtype="uint32",
            source_dtype="float32",
        )
    ) == _cvt_variant("f32", "tf32", _cvt_mode("PackedMode", "Rna", "SatFinite", "NoRelu"))
