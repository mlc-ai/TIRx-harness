from __future__ import annotations

import tvm
from tvm.backend.cuda.ptx.table import mods
from tvm.ir import Expr
from tvm.script import tirx as T
from tvm.tirx import Stmt
from tvm_ffi import structural_walk

from tests.numsim.support.manifest import emitted_module, emitted_calls
from tirx_harness.numsim.transpiler import native_frontend
from tirx_harness.numsim.transpiler.ptx_dialect import PTX_SCHEMA_BY_OP_NAME, decode_ptx_call

_TENSOR_OVERRIDE_SUFFIXES = (
    "override_address",
    "override_global_dim_b8",
    "override_global_dim_b16",
    "override_global_dim_stride_b8",
    "override_global_dim_stride_b16",
)
PTX_BULK_CACHE_HINT_CALLS = (
    "tirx.ptx.cp_async_bulk_prefetch",
    "tirx.ptx.cp_async_bulk_prefetch_evict_last",
    "tirx.ptx.applypriority_async_bulk",
)
PTX_TENSOR_CACHE_HINT_CALLS = (
    "tirx.ptx.cp_async_bulk_tensor_prefetch",
    "tirx.ptx.cp_async_bulk_tensor_prefetch_evict_last",
    "tirx.ptx.applypriority_async_bulk_tensor",
    "tirx.ptx.cp_async_bulk_tensor_prefetch_im2col",
    "tirx.ptx.cp_async_bulk_tensor_prefetch_im2col_evict_last",
    "tirx.ptx.applypriority_async_bulk_tensor_im2col",
    *(f"tirx.ptx.cp_async_bulk_tensor_prefetch_{suffix}" for suffix in _TENSOR_OVERRIDE_SUFFIXES),
    *(
        f"tirx.ptx.cp_async_bulk_tensor_prefetch_{suffix}"
        for suffix in (
            "override_address_evict_last",
            "override_global_dim_evict_last_b8",
            "override_global_dim_evict_last_b16",
            "override_global_dim_stride_evict_last_b8",
            "override_global_dim_stride_evict_last_b16",
        )
    ),
    *(f"tirx.ptx.applypriority_async_bulk_tensor_{suffix}" for suffix in _TENSOR_OVERRIDE_SUFFIXES),
    "tirx.ptx.cp_async_bulk_tensor_prefetch_override_address_im2col",
    "tirx.ptx.cp_async_bulk_tensor_prefetch_override_address_im2col_evict_last",
    "tirx.ptx.applypriority_async_bulk_tensor_override_address_im2col",
)
PTX_CACHE_HINT_CALLS = (
    "tirx.ptx.prefetch",
    "tirx.ptx.prefetch_valid_addr",
    "tirx.ptx.prefetchu",
    "tirx.ptx.applypriority",
    *PTX_BULK_CACHE_HINT_CALLS,
    *PTX_TENSOR_CACHE_HINT_CALLS,
)

_IM2COL_HINT_CALLS = frozenset(
    {
        "tirx.ptx.applypriority_async_bulk_tensor_im2col",
        "tirx.ptx.applypriority_async_bulk_tensor_override_address_im2col",
        "tirx.ptx.cp_async_bulk_tensor_prefetch_im2col_evict_last",
        "tirx.ptx.cp_async_bulk_tensor_prefetch_override_address_im2col_evict_last",
    }
)
_CANONICAL_TOKENS = {
    "tirx.ptx.applypriority": ("", "L2::evict_normal"),
    "tirx.ptx.applypriority_async_bulk": (
        "async",
        "bulk",
        "",
        "bulk_group",
        "L2::evict_normal",
    ),
    "tirx.ptx.applypriority_async_bulk_tensor": (
        "async",
        "bulk",
        "tensor",
        "2d",
        "",
        "bulk_group",
        "tile",
        "L2::evict_normal",
    ),
    "tirx.ptx.cp_async_bulk_prefetch_evict_last": (
        "async",
        "bulk",
        "prefetch",
        "L2",
        "global",
        "L2::evict_last",
    ),
    "tirx.ptx.cp_async_bulk_tensor_prefetch_evict_last": (
        "async",
        "bulk",
        "prefetch",
        "tensor",
        "2d",
        "L2",
        "global",
        "tile",
        "L2::evict_last",
    ),
    "tirx.ptx.prefetch_valid_addr": ("", "L1::32B", "valid_addr"),
    "tirx.ptx.prefetchu": ("L1",),
}


@T.prim_func
def baseline_cache_hint_forms(
    source: T.Buffer((128,), "uint8"),
    input_map: T.TensorMap(),
):
    T.device_entry()
    T.ptx["prefetch.global.L2"](source.ptr_to([0]))
    T.ptx["cp.async.bulk.prefetch.L2.global"](source.ptr_to([0]), T.uint32(32))
    T.ptx["cp.async.bulk.prefetch.tensor.2d.L2.global.tile"](T.address_of(input_map), 0, 0)


def _ptx_calls(function) -> dict[str, object]:
    calls = {}

    def visit(node: object) -> None:
        op_name = str(getattr(getattr(node, "op", None), "name", ""))
        if op_name.startswith("tirx.ptx."):
            assert op_name not in calls
            calls[op_name] = node

    structural_walk(function.body, ((Expr, Stmt), visit))
    return calls


def _public_form_call(op_name: str, tokens: tuple[str, ...], *, predicated: bool):
    entry = PTX_SCHEMA_BY_OP_NAME[op_name]
    modifier_map = mods(entry, tokens)
    spelling = ".".join((entry.family, *(token for token in tokens if token)))
    if op_name in PTX_TENSOR_CACHE_HINT_CALLS:
        rank = int(modifier_map["dim"].removesuffix("d"))
        coordinate_count = 5 if modifier_map["load_mode"] == "tile::gather4" else rank
        arguments = ["T.address_of(tmap)", *("T.int32(0)" for _ in range(coordinate_count))]
        if modifier_map.get("cache"):
            arguments.append("T.uint64(0)")
    elif op_name in PTX_BULK_CACHE_HINT_CALLS:
        arguments = ["source.ptr_to([0])", "T.uint32(32)"]
        if modifier_map.get("cache"):
            arguments.append("T.uint64(0)")
    elif op_name == "tirx.ptx.prefetch" and modifier_map["tensormap"]:
        arguments = ["T.address_of(tmap)"]
    else:
        arguments = ["source.ptr_to([0])"]
    if predicated:
        arguments.append("pred=T.bool(True)")
    function = tvm.script.from_source(
        "\n".join(
            (
                "@T.prim_func",
                "def kernel(source: T.Buffer((256,), 'uint8'), tmap: T.TensorMap()):",
                "    T.device_entry()",
                f'    T.ptx["{spelling}"]({", ".join(arguments)})',
            )
        ),
        {"T": T},
    )
    calls = _ptx_calls(function)
    assert tuple(calls) == (op_name,)
    return function, calls[op_name]


def test_cache_hint_operations_share_ordering_only_support_contract():
    specs = {spec["ir_name"]: spec for spec in native_frontend.registry_ops()}

    for op_name in PTX_CACHE_HINT_CALLS:
        spec = specs[op_name]
        assert spec["support"] == "ordering_only"
        assert spec["reason"]


def test_representative_cache_hint_forms_preserve_predication():
    for op_name, tokens in _CANONICAL_TOKENS.items():
        for predicated in (False, True):
            function, _call = _public_form_call(op_name, tokens, predicated=predicated)
            body = emitted_module(function)
            assert ("let mut ctx = ctx.with_active_mask(" in body) is predicated


def test_cache_hint_specializations_preserve_each_instruction_contract():
    calls = {}
    instructions = {}
    for op_name, tokens in _CANONICAL_TOKENS.items():
        function, call = _public_form_call(op_name, tokens, predicated=True)
        calls[op_name] = call
        instructions[op_name] = [emitted.head for emitted in emitted_calls(function, call)]

    assert instructions == {
        "tirx.ptx.applypriority": [
            "v2::async_copy::applypriority::<v2::async_copy::variant::ApplyPriority>"
        ],
        "tirx.ptx.applypriority_async_bulk": [
            "v2::async_copy::applypriority::<v2::async_copy::variant::BulkApplyPriority>"
        ],
        "tirx.ptx.applypriority_async_bulk_tensor": [
            "v2::async_copy::applypriority::<v2::async_copy::variant::TensorApplyPriority<2>>"
        ],
        "tirx.ptx.cp_async_bulk_prefetch_evict_last": ["v2::async_copy::cp_async_bulk_prefetch"],
        "tirx.ptx.cp_async_bulk_tensor_prefetch_evict_last": [
            "v2::async_copy::cp_async_bulk_prefetch_tensor::"
            "<v2::async_copy::variant::TensorPrefetchEvictLast<2>>"
        ],
        "tirx.ptx.prefetch_valid_addr": ["v2::async_copy::prefetch_valid_address"],
        "tirx.ptx.prefetchu": [],
    }

    applypriority = decode_ptx_call(calls["tirx.ptx.applypriority"])
    assert applypriority.operand("size") == ("128",)


def test_baseline_cache_hint_forms_lower_to_their_base_instructions():
    calls = _ptx_calls(baseline_cache_hint_forms)
    instructions = {
        op_name: [emitted.head for emitted in emitted_calls(baseline_cache_hint_forms, call)]
        for op_name, call in calls.items()
    }

    assert instructions == {
        "tirx.ptx.cp_async_bulk_prefetch": ["v2::async_copy::cp_async_bulk_prefetch"],
        "tirx.ptx.cp_async_bulk_tensor_prefetch": [
            "v2::async_copy::cp_async_bulk_prefetch_tensor::"
            "<v2::async_copy::variant::TensorPrefetch<2>>"
        ],
        "tirx.ptx.prefetch": [],
    }


def test_baseline_cache_hint_forms_preserve_target_predication():
    canonical_forms = {
        "tirx.ptx.prefetch": ("global", "L2", "", ""),
        "tirx.ptx.cp_async_bulk_prefetch": (
            "async",
            "bulk",
            "prefetch",
            "L2",
            "global",
            "",
        ),
        "tirx.ptx.cp_async_bulk_tensor_prefetch": (
            "async",
            "bulk",
            "prefetch",
            "tensor",
            "2d",
            "L2",
            "global",
            "tile",
            "",
        ),
    }
    for op_name, tokens in canonical_forms.items():
        function, _call = _public_form_call(op_name, tokens, predicated=True)
        assert "let mut ctx = ctx.with_active_mask(" in emitted_module(function)
