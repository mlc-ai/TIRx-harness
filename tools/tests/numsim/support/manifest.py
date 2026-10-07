"""Manifest-side views of one kernel for tests that no longer read the Python resolver.

The frontend records every rejected call as an ``op#<id>:...`` entry of
``unsupported`` (raised by ``verify``) and every unmodeled form as
``UnmodeledTIRxFormError`` from ``analyze`` itself; an accepted call's
specialization is the engine instruction its site emits.  The helpers below
expose exactly those facts, plus the complete native module text, so a test never
needs a Python-side resolution object.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from typing import Any, NamedTuple

import tvm
from tvm import tirx
from tvm.ir import Expr
from tvm.script import tirx as T
from tvm.tirx import Stmt
from tvm_ffi import structural_map

from tirx_harness.numsim.transpiler.artifact_template import emit_rust_module
from tirx_harness.numsim.transpiler.frontend import PrimFuncSpec, analyze, verify
from tirx_harness.numsim.transpiler.host_prelude import normalize_host_tensor_map_prelude

DEVICE_ENTRY_ATTR = "tirx.device_entry"


def parse_kernel(source: str) -> tirx.PrimFunc:
    """Parse one TVMScript kernel source."""

    return tvm.script.from_source(source, {"T": T})


def device_kernel(body: Stmt, params: Sequence[tirx.Var] = ()) -> tirx.PrimFunc:
    """A device-entry PrimFunc around ``body`` (what ``T.device_entry()`` parses to)."""

    entry = tirx.AttrStmt(0, DEVICE_ENTRY_ATTR, tirx.IntImm("bool", 1), body)
    return tirx.PrimFunc(list(params), entry)


def evaluated_kernel(call: Any, params: Sequence[tirx.Var] = ()) -> tirx.PrimFunc:
    """A device-entry PrimFunc whose only statement evaluates ``call``."""

    return device_kernel(tirx.Evaluate(call), params)


def _matcher(matches: str | Callable[[str], bool]) -> Callable[[str], bool]:
    if isinstance(matches, str):
        return lambda name: name == matches
    return matches


def replace_call(func: tirx.PrimFunc, original: Any, replacement: Any) -> tirx.PrimFunc:
    """``func`` with the exact ``original`` call node replaced by ``replacement``."""

    body = structural_map(
        func.body,
        ((Expr, Stmt), lambda node: replacement if node.same_as(original) else node),
    )
    return tirx.PrimFunc(func.params, body, func.ret_type, func.attrs)


def kernel_manifest(func: Any) -> PrimFuncSpec:
    """The kernel's manifest, rejected calls included in ``unsupported``."""

    return analyze(func).kernels[0]


def resolved_kernel(func: Any) -> PrimFuncSpec:
    """The kernel's manifest, raising ``UnsupportedTIRxError`` for any rejected call."""

    spec = analyze(func)
    verify(spec)
    return spec.kernels[0]


def _op_name(entry: Any) -> str:
    return str(getattr(entry.node.op, "name", ""))


def call_op_names(kernel: PrimFuncSpec) -> set[str]:
    """The op names of the kernel's calls."""

    return {_op_name(entry) for entry in kernel.source_map if entry.kind == "Call"}


class EmittedCall(NamedTuple):
    """One engine instruction a call site emits: ``function::<generics>(...)``."""

    function: str
    generics: str

    @property
    def head(self) -> str:
        return f"{self.function}::<{self.generics}>" if self.generics else self.function


_SITE = re.compile(r"v2::SiteId::new\((\d+)_u64\)")


def _enclosing_call(text: str, index: int) -> EmittedCall:
    """The call whose argument list holds ``text[index]``."""

    depth = 0
    open_paren = index - 1
    while text[open_paren] != "(" or depth:
        depth += {")": 1, "(": -1}.get(text[open_paren], 0)
        open_paren -= 1
    name_end, generics = open_paren, ""
    if text[open_paren - 1] == ">":
        depth, start = 0, open_paren - 1
        while True:
            depth += {">": 1, "<": -1}.get(text[start], 0)
            if not depth:
                break
            start -= 1
        assert text[start - 2 : start] == "::", text[start - 80 : open_paren]
        name_end, generics = start - 2, text[start + 1 : open_paren - 1]
    function = re.search(r"[A-Za-z_][A-Za-z0-9_:]*$", text[:name_end])
    assert function is not None, text[name_end - 80 : open_paren]
    return EmittedCall(function.group(0), generics)


def emitted_calls(
    func: tirx.PrimFunc, matches: str | Callable[[str], bool] | Any
) -> list[EmittedCall]:
    """The ``v2::`` instructions emitted at the calls ``matches`` selects.

    ``matches`` is an op name, a predicate over op names, or one exact call node.
    Calls follow their order in the complete emitted module.
    """

    func = normalize_host_tensor_map_prelude(func)
    spec = analyze(func)
    verify(spec)
    kernel = spec.kernels[0]
    if isinstance(matches, Expr):
        selected = lambda entry: entry.node.same_as(matches)  # noqa: E731
    else:
        accept = _matcher(matches)
        selected = lambda entry: accept(_op_name(entry))  # noqa: E731
    sites = {entry.op_id for entry in kernel.source_map if entry.kind == "Call" and selected(entry)}
    assert sites, "no call matches"
    text = emit_rust_module(spec, func)
    calls = []
    for site in _SITE.finditer(text):
        if int(site.group(1)) in sites:
            call = _enclosing_call(text, site.start())
            if call.function.startswith("v2::"):
                calls.append(call)
    return calls


def emitted_module(func: tirx.PrimFunc) -> str:
    """The complete native plain-mode module containing one kernel."""

    func = normalize_host_tensor_map_prelude(func)
    return emit_rust_module(analyze(func), func)


__all__ = [
    "DEVICE_ENTRY_ATTR",
    "EmittedCall",
    "call_op_names",
    "device_kernel",
    "emitted_module",
    "emitted_calls",
    "evaluated_kernel",
    "kernel_manifest",
    "parse_kernel",
    "replace_call",
    "resolved_kernel",
]
