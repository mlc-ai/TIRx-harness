from __future__ import annotations

from dataclasses import dataclass

import ml_dtypes
import numpy as np
import pytest
import tvm
from tvm.backend.cuda.ptx.table import (
    canonical_dtypes,
    mods,
    operand_type,
    variants,
)
from tvm.script import tirx as T

from tests.numsim.support.execution import run_checked
from tirx_harness import numsim, racecheck, synccheck
from tirx_harness.numsim.errors import NumSimExecutionError
from tirx_harness.numsim.transpiler.ptx_dialect import PTX_SCHEMA_BY_OP_NAME

PTX_BIT_OP_CALLS = (
    "tirx.ptx.abs",
    "tirx.ptx.and",
    "tirx.ptx.or",
    "tirx.ptx.xor",
    "tirx.ptx.selp",
    "tirx.ptx.bfe",
    "tirx.ptx.bfi",
    "tirx.ptx.bfind",
    "tirx.ptx.bmsk",
    "tirx.ptx.brev",
    "tirx.ptx.clz",
    "tirx.ptx.cnot",
    "tirx.ptx.not",
    "tirx.ptx.popc",
    "tirx.ptx.shf",
    "tirx.ptx.szext",
    "tirx.ptx.shl",
    "tirx.ptx.shr",
    "tirx.ptx.prmt",
    "tirx.ptx.fns",
)


@dataclass(frozen=True)
class _RuntimeForm:
    op_name: str
    spelling: str
    modifiers: tuple[tuple[str, str], ...]
    dtypes: tuple[str, ...]
    predicate_sources: frozenset[int]
    output_row: int


def _runtime_forms() -> tuple[_RuntimeForm, ...]:
    rows: dict[str, int] = {}
    forms = []
    for op_name in PTX_BIT_OP_CALLS:
        entry = PTX_SCHEMA_BY_OP_NAME[op_name]
        for tokens in variants(entry):
            modifier_map = mods(entry, tokens)
            dtypes = canonical_dtypes(entry, tokens)
            destination_dtype = dtypes[0]
            output_row = rows.get(destination_dtype, 0)
            rows[destination_dtype] = output_row + 1
            forms.append(
                _RuntimeForm(
                    op_name=op_name,
                    spelling=".".join((entry.ptx_name, *(token for token in tokens if token))),
                    modifiers=tuple(modifier_map.items()),
                    dtypes=dtypes,
                    predicate_sources=frozenset(
                        index - 1
                        for index, slot in enumerate(entry.typed_operands)
                        if index > 0 and operand_type(slot, modifier_map) == "pred"
                    ),
                    output_row=output_row,
                )
            )
    return tuple(forms)


_RUNTIME_FORMS = _runtime_forms()


def _make_all_forms_kernel():
    inputs = sorted(
        {(index, dtype) for form in _RUNTIME_FORMS for index, dtype in enumerate(form.dtypes[1:])}
    )
    output_rows: dict[str, int] = {}
    for form in _RUNTIME_FORMS:
        output_rows[form.dtypes[0]] = max(output_rows.get(form.dtypes[0], 0), form.output_row + 1)

    parameters = [
        *(f'input_{index}_{dtype}: T.Buffer((32,), "{dtype}")' for index, dtype in inputs),
        *(
            f'output_{dtype}: T.Buffer(({4 * rows}, 32), "{dtype}")'
            for dtype, rows in sorted(output_rows.items())
        ),
    ]
    lines = [
        "@T.prim_func",
        "def ptx_bitops_all_forms(",
        *(f"    {parameter}," for parameter in parameters),
        "):",
        "    T.device_entry()",
        "    _warp = T.warp_id([1])",
        "    lane = T.lane_id([32])",
        "    masked = T.Select(lane % 2 == 0, lane, 32)",
        *(f'    dst_{dtype} = T.alloc_local((1,), "{dtype}")' for dtype in sorted(output_rows)),
    ]
    for form in _RUNTIME_FORMS:
        dtype, row = form.dtypes[0], 4 * form.output_row
        for position, index, predicate in (
            (0, "lane", ""),
            (1, "masked", ", pred=lane % 2 == 0, preserve_dst=True"),
            (2, "masked", ", pred=lane % 2 == 0"),
            (3, "32", ", pred=False, preserve_dst=True"),
        ):
            arguments = [f"dst_{dtype}[0]"]
            arguments.extend(
                (
                    f"T.ptx.pred(input_{source}_{carrier}[{index}])"
                    if source in form.predicate_sources
                    else f"input_{source}_{carrier}[{index}]"
                )
                for source, carrier in enumerate(form.dtypes[1:])
            )
            if form.op_name in {"tirx.ptx.bfe", "tirx.ptx.bfi"}:
                # PTX restricts both controls to 0..255. Wide raw inputs are
                # tested as errors separately, not as a portable GPU oracle.
                for argument in range(len(arguments) - 2, len(arguments)):
                    arguments[argument] = f"({arguments[argument]} & T.uint32(255))"
            if form.op_name == "tirx.ptx.fns":
                # PTX leaves bases outside 0..31 undefined.
                arguments[2] = f"({arguments[2]} & T.uint32(31))"
            lines += [
                f'    dst_{dtype}[0] = T.cast(85, "{dtype}")',
                f'    T.ptx["{form.spelling}"]({", ".join(arguments)}{predicate})',
                f"    output_{dtype}[{row + position}, lane] = dst_{dtype}[0]",
            ]
    return tvm.script.from_source("\n".join(lines), {"T": T})


def _unsigned_words(width: int) -> np.ndarray:
    mask = (1 << width) - 1
    edges = [
        0,
        1,
        2,
        3,
        1 << (width - 1),
        (1 << (width - 1)) - 1,
        mask,
        mask - 1,
        0x5555_5555_5555_5555 & mask,
        0xAAAA_AAAA_AAAA_AAAA & mask,
    ]
    values = [
        *edges,
        *((0x9E37_79B9_7F4A_7C15 * lane + 0xD1B5_4A32_D192_ED03) & mask for lane in range(10, 32)),
    ]
    return np.asarray(values, dtype=getattr(np, f"uint{width}"))


_CONTROL_WORDS = np.asarray(
    [
        0,
        1,
        2,
        7,
        15,
        16,
        31,
        32,
        33,
        63,
        64,
        65,
        127,
        255,
        256,
        257,
        0xFFFF_FFFF,
        *((37 * lane + 11) & 0xFFFF_FFFF for lane in range(17, 32)),
    ],
    dtype=np.uint32,
)


def _input_array(index: int, dtype: str) -> np.ndarray:
    if (index, dtype) == (2, "int32"):
        return np.asarray(
            [0, 1, -1, 2, -2, 3, -3, 31, -31, 32, -32, 33, -33, -(1 << 31), (1 << 31) - 1, 4] * 2,
            dtype=np.int32,
        )
    width = np.dtype(dtype).itemsize * 8
    words = _CONTROL_WORDS.copy() if dtype == "uint32" and index > 0 else _unsigned_words(width)
    return words.view(dtype)


def _low_mask(width: int) -> int:
    return (1 << width) - 1


def _signed(bits: int, width: int) -> int:
    return bits - (1 << width) if bits & (1 << (width - 1)) else bits


def _bfe(value: int, position: int, length: int, width: int, signed: bool) -> int:
    value &= _low_mask(width)
    position &= 0xFF
    length &= 0xFF
    if length == 0:
        return 0
    copied = 0 if position >= width else min(length, width - position)
    copied_mask = _low_mask(copied)
    result = 0 if copied == 0 else (value >> position) & copied_mask
    sign_position = min(position + length - 1, width - 1)
    if signed and value & (1 << sign_position):
        result |= _low_mask(width) ^ copied_mask
    return result & _low_mask(width)


def _bfi(source: int, base: int, position: int, length: int, width: int) -> int:
    mask = _low_mask(width)
    source &= mask
    base &= mask
    position &= 0xFF
    length &= 0xFF
    copied = 0 if position >= width else min(length, width - position)
    source_mask = _low_mask(copied)
    insertion_mask = source_mask << position
    return (base & ~insertion_mask) | ((source & source_mask) << position)


def _bfind(value: int, width: int, signed: bool, shift_amount: bool) -> int:
    mask = _low_mask(width)
    bits = value & mask
    if signed and bits & (1 << (width - 1)):
        bits = (~bits) & mask
    if bits == 0:
        return 0xFFFF_FFFF
    position = bits.bit_length() - 1
    return width - 1 - position if shift_amount else position


def _bmsk(position: int, width: int, clamp: bool) -> int:
    if clamp:
        if position >= 32:
            return 0
        effective_position = position
        effective_width = min(width, 32)
    else:
        effective_position = position & 31
        effective_width = width & 31
    effective_width = min(effective_width, 32 - effective_position)
    return _low_mask(effective_width) << effective_position


def _oracle(form: _RuntimeForm, sources: tuple[int, ...]) -> int:
    modifiers = dict(form.modifiers)
    ptx_type = modifiers["type"]
    width = 1 if ptx_type == "pred" else int(ptx_type[1:])
    signed = ptx_type.startswith("s")
    op = form.op_name.rsplit(".", 1)[-1]
    if op == "selp":
        return sources[0] if sources[2] else sources[1]
    if op == "abs":
        return _signed((-sources[0] if sources[0] < 0 else sources[0]) & _low_mask(width), width)
    if op == "bfe":
        bits = _bfe(*sources, width, signed)
        return _signed(bits, width) if signed else bits
    if op == "bfi":
        return _bfi(*sources, width)
    if op == "bfind":
        return _bfind(sources[0], width, signed, bool(modifiers["shiftamt"]))
    if op == "bmsk":
        return _bmsk(*sources, modifiers["mode"] == "clamp")
    if op == "brev":
        bits = sources[0] & _low_mask(width)
        return int(f"{bits:0{width}b}"[::-1], 2)
    if op == "clz":
        bits = sources[0] & _low_mask(width)
        return width if bits == 0 else width - bits.bit_length()
    if op == "cnot":
        return int((sources[0] & _low_mask(width)) == 0)
    if op == "not":
        if ptx_type == "pred":
            return int(sources[0] == 0)
        return (~sources[0]) & _low_mask(width)
    if op in {"and", "or", "xor"}:
        a, b = map(bool, sources) if ptx_type == "pred" else sources
        return {"and": a & b, "or": a | b, "xor": a ^ b}[op]
    if op == "popc":
        return (sources[0] & _low_mask(width)).bit_count()
    if op in {"shl", "shr"}:
        value, shift = sources
        shift = min(shift, width)
        if op == "shl":
            return (value << shift) & _low_mask(width)
        return value >> shift
    if op == "prmt":
        a, b, control = sources
        # Independent literal ISA rows, listed least-significant byte first.
        rows = {
            "f4e": ("0123", "1234", "2345", "3456"),
            "b4e": ("0765", "1076", "2107", "3210"),
            "rc8": ("0000", "1111", "2222", "3333"),
            "ecl": ("0123", "1123", "2223", "3333"),
            "ecr": ("0000", "0111", "0122", "0123"),
            "rc16": ("0101", "2323", "0101", "2323"),
        }
        mode = modifiers["mode"]
        selectors = (
            [int(byte) for byte in rows[mode][control & 3]]
            if mode
            else [(control >> (4 * byte)) & 15 for byte in range(4)]
        )
        source = a | (b << 32)
        result = 0
        for byte, selector in enumerate(selectors):
            value = (source >> (8 * (selector & 7))) & 255
            if selector & 8:
                value = 255 if value >= 128 else 0
            result |= value << (8 * byte)
        return result
    if op == "fns":
        mask, base, offset = sources
        base &= 31
        if offset == 0:
            return base if mask & (1 << base) else 0xFFFF_FFFF
        candidates = [bit for bit in range(32) if mask & (1 << bit)]
        candidates = (
            [bit for bit in candidates if bit >= base]
            if offset > 0
            else [bit for bit in reversed(candidates) if bit <= base]
        )
        return candidates[abs(offset) - 1] if abs(offset) <= len(candidates) else 0xFFFF_FFFF
    if op == "shf":
        low, high, shift = sources
        shift = min(shift, 32) if modifiers["mode"] == "clamp" else shift & 31
        joined = ((high & 0xFFFF_FFFF) << 32) | (low & 0xFFFF_FFFF)
        if modifiers["dir"] == "l":
            return ((joined << shift) >> 32) & 0xFFFF_FFFF
        return (joined >> shift) & 0xFFFF_FFFF
    if op == "szext":
        value, source_width = sources
        if modifiers["mode"] == "clamp" and source_width >= 32:
            return value
        source_width &= 31
        if source_width == 0:
            return 0
        bits = value & _low_mask(source_width)
        if signed and bits & (1 << (source_width - 1)):
            return bits - (1 << source_width)
        return bits
    raise AssertionError(f"missing bit-operation oracle for {form.op_name}")


def _bitop_case():
    inputs = {
        (index, dtype): _input_array(index, dtype)
        for form in _RUNTIME_FORMS
        for index, dtype in enumerate(form.dtypes[1:])
    }
    output_rows: dict[str, int] = {}
    for form in _RUNTIME_FORMS:
        output_rows[form.dtypes[0]] = max(output_rows.get(form.dtypes[0], 0), form.output_row + 1)
    arguments = {
        **{f"input_{index}_{dtype}": value for (index, dtype), value in inputs.items()},
        **{
            f"output_{dtype}": np.zeros((4 * rows, 32), dtype=getattr(np, dtype))
            for dtype, rows in output_rows.items()
        },
    }
    expected = {
        dtype: np.full((4 * rows, 32), 85, dtype=getattr(np, dtype))
        for dtype, rows in output_rows.items()
    }
    for form in _RUNTIME_FORMS:
        source_arrays = tuple(inputs[index, dtype] for index, dtype in enumerate(form.dtypes[1:]))
        if dict(form.modifiers)["type"] == "pred":
            # The target .pred bridge retains Boolean truth, not carrier 85.
            expected[form.dtypes[0]][4 * form.output_row : 4 * form.output_row + 4] = 1
        values = np.asarray(
            [
                _oracle(
                    form,
                    tuple(
                        array[lane] if form.op_name.endswith(".selp") else int(array[lane])
                        for array in source_arrays
                    ),
                )
                for lane in range(32)
            ],
            dtype=getattr(np, form.dtypes[0]),
        )
        expected[form.dtypes[0]][4 * form.output_row] = values
        expected[form.dtypes[0]][4 * form.output_row + 1, ::2] = values[::2]
        expected[form.dtypes[0]][4 * form.output_row + 2, ::2] = values[::2]
    return (
        _make_all_forms_kernel(),
        arguments,
        expected,
    )


def _check_outputs(outputs, expected):
    for form in _RUNTIME_FORMS:
        dtype = form.dtypes[0]
        for position in range(4):
            lanes = slice(None, None, 2) if position == 2 else slice(None)
            row = 4 * form.output_row + position
            raw = f"uint{np.dtype(dtype).itemsize * 8}"
            np.testing.assert_array_equal(
                outputs[f"output_{dtype}"][row].view(raw)[lanes],
                expected[dtype][row].view(raw)[lanes],
                err_msg=f"{form.spelling}, predicate mode {position}",
            )


def test_ptx_bitop_forms_and_predicates_match_independent_oracle(tmp_path):
    kernel, arguments, expected = _bitop_case()
    result = run_checked(kernel, arguments, cache_dir=tmp_path)
    _check_outputs(result.outputs, expected)


def test_bitfield_control_domain_is_checked_only_on_issuing_lanes(tmp_path):
    for spelling in ("bfe.u32", "bfe.s32", "bfe.u64", "bfe.s64", "bfi.b32", "bfi.b64"):
        bits = int(spelling[-2:])
        dtype = f"{'int' if spelling.startswith('bfe.s') else 'uint'}{bits}"
        data = f"T.{dtype}(1), " + (f"T.{dtype}(0), " if spelling.startswith("bfi") else "")
        kernel = tvm.script.from_source(
            f'''@T.prim_func
def kernel(position: T.Buffer((32,), "uint32"), length: T.Buffer((32,), "uint32"),
           output: T.Buffer((32,), "{dtype}")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    output[lane] = T.{dtype}(85)
    T.ptx["{spelling}"](output[lane], {data}position[lane], length[lane],
                      pred=lane % 2 == 0, preserve_dst=True)
''',
            {"T": T},
        )
        inputs = dict(
            position=np.zeros(32, np.uint32),
            length=np.ones(32, np.uint32),
            output=np.zeros(32, dtype),
        )
        inputs["position"][1::2] = 0xFFFFFFFF
        inputs["length"][1::2] = 0xFFFFFFFF
        result = run_checked(kernel, inputs, cache_dir=tmp_path)
        # A signed one-bit field containing 1 sign-extends to -1.
        extracted = -1 if spelling.startswith("bfe.s") else 1
        np.testing.assert_array_equal(result.outputs["output"], [extracted, 85] * 16)
        artifact = numsim.transpile(kernel, cache_dir=tmp_path)
        for control in ("position", "length"):
            for invalid in (256, 0xFFFFFFFF):
                original = inputs[control][0]
                inputs[control][0] = invalid
                with pytest.raises(NumSimExecutionError, match="outside the defined 0..255 range"):
                    numsim.Engine().run(artifact, inputs)
                inputs[control][0] = original


def test_fns_base_domain_is_checked_only_on_issuing_lanes(tmp_path):
    @T.prim_func
    def kernel(bases: T.Buffer((32,), "uint32"), output: T.Buffer((32,), "uint32")):
        T.device_entry()
        _warp = T.warp_id([1])
        lane = T.lane_id([32])
        output[lane] = T.uint32(85)
        T.ptx.fns.b32(
            output[lane],
            T.uint32(0xFFFFFFFF),
            bases[lane],
            T.int32(-1),
            pred=lane % 2 == 0,
            preserve_dst=True,
        )

    inputs = dict(bases=np.arange(32, dtype=np.uint32), output=np.zeros(32, np.uint32))
    inputs["bases"][1::2] = 0xFFFFFFFF
    for checker in (synccheck, racecheck):
        checker(kernel, inputs).require_clean()
    artifact = numsim.transpile(kernel, cache_dir=tmp_path)
    expected = np.arange(32, dtype=np.uint32)
    expected[1::2] = 85
    np.testing.assert_array_equal(numsim.Engine().run(artifact, inputs).outputs["output"], expected)
    for invalid in (32, 0xFFFFFFFF):
        inputs["bases"][0] = invalid
        with pytest.raises(
            NumSimExecutionError, match="fns.b32 base is outside the defined 0..31 range"
        ):
            numsim.Engine().run(artifact, inputs)


__all__ = []


_B16_DTYPES = ("uint16", "int16", "float16", "bfloat16")


def _make_b16_carrier_kernel():
    lines = [
        "@T.prim_func",
        "def ptx_b16_carriers(",
        *(f'    input_{dtype}: T.Buffer((32,), "{dtype}"),' for dtype in _B16_DTYPES),
        *(f'    output_{dtype}: T.Buffer((8, 32), "{dtype}"),' for dtype in _B16_DTYPES),
        "):",
        "    T.device_entry()",
        "    _warp = T.warp_id([1])",
        "    lane = T.lane_id([32])",
    ]
    for destination_dtype in _B16_DTYPES:
        row = 0
        for instruction in ("not", "cnot"):
            for source_dtype in _B16_DTYPES:
                lines.append(
                    f'    T.ptx["{instruction}.b16"]('
                    f"output_{destination_dtype}[{row}, lane], input_{source_dtype}[lane])"
                )
                row += 1
    return tvm.script.from_source("\n".join(lines), {"T": T})


def _view_b16(bits: np.ndarray, dtype: str) -> np.ndarray:
    return bits.view(
        {
            "uint16": np.uint16,
            "int16": np.int16,
            "float16": np.float16,
            "bfloat16": ml_dtypes.bfloat16,
        }[dtype]
    )


def test_b16_carrier_cross_product_preserves_every_payload_bit(tmp_path):
    edge_bits = np.asarray(
        [
            0x0000,
            0x0001,
            0x03FF,
            0x0400,
            0x3C00,
            0x7BFF,
            0x7C00,
            0x7C01,
            0x7DFF,
            0x7E00,
            0x7FFF,
            0x8000,
            0xFC00,
            0xFC01,
            0xFE00,
            0xFFFF,
            *((0x9E37 * lane + 0x1234) & 0xFFFF for lane in range(16, 32)),
        ],
        dtype=np.uint16,
    )
    source_bits = {
        dtype: np.bitwise_xor(edge_bits, np.uint16(index * 0x1111))
        for index, dtype in enumerate(_B16_DTYPES)
    }
    arguments = {
        **{f"input_{dtype}": _view_b16(bits, dtype) for dtype, bits in source_bits.items()},
        **{
            f"output_{dtype}": _view_b16(
                np.full((8, 32), np.uint16(0xDEAD), dtype=np.uint16), dtype
            )
            for dtype in _B16_DTYPES
        },
    }
    result = numsim.Engine().run(
        numsim.transpile(_make_b16_carrier_kernel(), cache_dir=tmp_path),
        arguments,
        outputs=tuple(f"output_{dtype}" for dtype in _B16_DTYPES),
    )
    expected = np.stack(
        [
            *(np.bitwise_not(source_bits[dtype]) for dtype in _B16_DTYPES),
            *((source_bits[dtype] == 0).astype(np.uint16) for dtype in _B16_DTYPES),
        ]
    )
    for dtype in _B16_DTYPES:
        np.testing.assert_array_equal(result.outputs[f"output_{dtype}"].view(np.uint16), expected)
