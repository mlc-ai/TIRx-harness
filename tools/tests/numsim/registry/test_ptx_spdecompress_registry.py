from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import pytest
import tvm
from tvm.backend.cuda.ptx.table import TABLE, mods, operand_layout
from tvm.script import tirx as T

from tests.numsim.support.manifest import emitted_calls, resolved_kernel
from tirx_harness.numsim.errors import UnsupportedTIRxError
from tirx_harness.numsim.transpiler.frontend import analyze
from tirx_harness.numsim.transpiler.artifact_template import emit_rust_module


@dataclass(frozen=True)
class _Form:
    elem_bits: int
    index_bits: int
    src: int
    dst: int
    num: int
    data_registers: int
    metadata_registers: int
    compressed_registers: int

    @property
    def spelling(self) -> str:
        return (
            f"spdecompress.b{self.elem_bits}.b{self.index_bits}."
            f"sp::{self.src}:{self.dst}.x{self.num}"
        )


def _form(spelling: str) -> _Form:
    entry = TABLE["spdecompress"]
    modifiers = mods(entry, spelling.split("."))
    assert entry.check(modifiers) is None
    lanes = {operand.name: count for operand, _offset, count in operand_layout(entry, modifiers)}
    src, dst = map(int, modifiers["spfactor"].removeprefix("sp::").split(":"))
    return _Form(
        elem_bits=int(modifiers["elemsize"].removeprefix("b")),
        index_bits=int(modifiers["idxsize"].removeprefix("b")),
        src=src,
        dst=dst,
        num=int(modifiers["num"].removeprefix("x")),
        data_registers=lanes["data"],
        metadata_registers=lanes["mdata"],
        compressed_registers=lanes["cdata"],
    )


def _defined_forms() -> tuple[_Form, ...]:
    forms = []
    entry = TABLE["spdecompress"]
    for spelling in product(*(slot.choices for slot in entry.slots)):
        modifiers = mods(entry, spelling)
        if entry.check(modifiers) is not None:
            continue
        lanes = {
            operand.name: count for operand, _offset, count in operand_layout(entry, modifiers)
        }
        src, dst = map(int, modifiers["spfactor"].removeprefix("sp::").split(":"))
        forms.append(
            _Form(
                elem_bits=int(modifiers["elemsize"].removeprefix("b")),
                index_bits=int(modifiers["idxsize"].removeprefix("b")),
                src=src,
                dst=dst,
                num=int(modifiers["num"].removeprefix("x")),
                data_registers=lanes["data"],
                metadata_registers=lanes["mdata"],
                compressed_registers=lanes["cdata"],
            )
        )
    return tuple(forms)


_DEFINED_FORMS = _defined_forms()


@T.prim_func
def spdecompress_aliased_register_views():
    T.device_entry()
    storage = T.alloc_buffer((3,), "uint32", scope="local")
    alias = T.decl_buffer((2,), "uint32", data=storage.data, elem_offset=1, scope="local")
    T.ptx["spdecompress.b8.b4.sp::1:2.x2"](storage[1], alias[0], storage[2])


@T.prim_func
def spdecompress_disjoint_aliased_register_views():
    T.device_entry()
    storage = T.alloc_buffer((3,), "uint32", scope="local")
    alias = T.decl_buffer((2,), "uint32", data=storage.data, elem_offset=1, scope="local")
    T.ptx["spdecompress.b8.b4.sp::1:2.x2"](storage[0], alias[0], storage[2])


@T.prim_func
def spdecompress_dynamic_register_index(index: T.int32):
    T.device_entry()
    storage = T.alloc_buffer((3,), "uint32", scope="local")
    T.ptx["spdecompress.b8.b4.sp::1:2.x2"](storage[index], storage[1], storage[2])


def _kernel(form: _Form, dtypes: tuple[str, ...] | None = None, *, overlap: bool = False):
    argument_count = form.data_registers + form.metadata_registers + form.compressed_registers
    dtypes = dtypes or ("uint32",) * argument_count
    assert len(dtypes) == argument_count
    declarations = "\n".join(
        f'    operand_{index} = T.local_scalar("{dtype}")' for index, dtype in enumerate(dtypes)
    )
    arguments = ", ".join(
        "operand_0" if overlap else f"operand_{index}" for index in range(argument_count)
    )
    return tvm.script.from_source(
        f'''
@T.prim_func
def kernel():
    T.device_entry()
{declarations}
    T.ptx["{form.spelling}"]({arguments})
''',
        {"T": T},
    )


@pytest.mark.parametrize("form", _DEFINED_FORMS, ids=lambda form: form.spelling)
def test_every_defined_spdecompress_schema_form_resolves(form: _Form):
    assert [call.head for call in emitted_calls(_kernel(form), "tirx.ptx.spdecompress")] == [
        f"v2::reg::spdecompress::<v2::reg::variant::SpDecompress<"
        f"{form.elem_bits}, {form.index_bits}, {form.src}, {form.dst}, {form.num}>>"
    ]


def test_spdecompress_accepts_each_b32_carrier_without_changing_static_shape():
    form = _form("b8.b4.sp::2:4.x2")
    dtypes = (
        ("uint32",) * form.data_registers
        + ("int32",) * form.metadata_registers
        + ("float32",) * form.compressed_registers
    )
    assert emitted_calls(_kernel(form, dtypes), "tirx.ptx.spdecompress") == emitted_calls(
        _kernel(form), "tirx.ptx.spdecompress"
    )


def test_spdecompress_rejects_mixed_carriers_within_one_register_vector():
    form = _form("b8.b4.sp::2:4.x2")
    argument_count = form.data_registers + form.metadata_registers + form.compressed_registers
    mixed = ("uint32", "int32") + ("uint32",) * (argument_count - 2)
    with pytest.raises(tvm.error.DiagnosticError, match="spdecompress"):
        _kernel(form, mixed)


def test_spdecompress_rejects_statically_visible_register_overlap():
    form = _form("b8.b4.sp::1:2.x2")
    with pytest.raises(UnsupportedTIRxError, match="undefined register overlap"):
        resolved_kernel(_kernel(form, overlap=True))


def test_spdecompress_rejects_same_physical_register_through_alias_views():
    with pytest.raises(UnsupportedTIRxError, match="undefined register overlap.*aliased"):
        emit_rust_module(
            analyze(spdecompress_aliased_register_views), spdecompress_aliased_register_views
        )


def test_spdecompress_accepts_disjoint_registers_through_alias_views():
    emit_rust_module(
        analyze(spdecompress_disjoint_aliased_register_views),
        spdecompress_disjoint_aliased_register_views,
    )


def test_spdecompress_rejects_unknown_dynamic_register_overlap():
    with pytest.raises(UnsupportedTIRxError, match="cannot prove disjoint physical register"):
        emit_rust_module(
            analyze(spdecompress_dynamic_register_index),
            spdecompress_dynamic_register_index,
        )
