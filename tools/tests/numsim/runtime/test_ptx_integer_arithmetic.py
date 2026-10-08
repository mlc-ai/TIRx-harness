from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest
import tvm

from tirx_harness import numsim
from tvm.script import tirx as T


_INTEGER_TYPES = ("u16", "u32", "u64", "s16", "s32", "s64")
_WIDE_TYPES = ("u16", "s16", "u32", "s32")


def _dtype(ptx_type: str) -> str:
    return f"{'int' if ptx_type.startswith('s') else 'uint'}{ptx_type[1:]}"


@dataclass(frozen=True)
class _Form:
    op: str
    spelling: str
    ptx_type: str = ""
    mode: str = ""
    sat: bool = False
    atype: str = ""
    btype: str = ""

    @property
    def destination_dtype(self) -> str:
        if self.op in {"mul_wide", "mad_wide"}:
            bits = int(self.ptx_type[1:]) * 2
            return f"{'int' if self.ptx_type.startswith('s') else 'uint'}{bits}"
        if self.op in {"dp2a", "dp4a"}:
            return "uint32" if self.atype == self.btype == "u32" else "int32"
        return _dtype(self.ptx_type)

    @property
    def source_dtypes(self) -> tuple[str, ...]:
        if self.op == "neg_int":
            return (_dtype(self.ptx_type),)
        if self.op in {"div", "rem", "mul24", "mul_wide"}:
            return (_dtype(self.ptx_type),) * 2
        if self.op in {"sad", "mad24"}:
            return (_dtype(self.ptx_type),) * 3
        if self.op == "mad_wide":
            return (_dtype(self.ptx_type), _dtype(self.ptx_type), self.destination_dtype)
        return (_dtype(self.atype), _dtype(self.btype), self.destination_dtype)


_FORMS = (
    *(
        _Form(op, f"{op}.{ptx_type}", ptx_type=ptx_type)
        for op in ("div", "rem", "sad")
        for ptx_type in _INTEGER_TYPES
    ),
    *(
        _Form("mul24", f"mul24.{mode}.{ptx_type}", ptx_type=ptx_type, mode=mode)
        for mode in ("hi", "lo")
        for ptx_type in ("u32", "s32")
    ),
    *(
        _Form("mul_wide", f"mul.wide.{ptx_type}", ptx_type=ptx_type, mode="wide")
        for ptx_type in _WIDE_TYPES
    ),
    _Form("mad24", "mad24.hi.sat.s32", ptx_type="s32", mode="hi", sat=True),
    *(
        _Form("mad24", f"mad24.{mode}.{ptx_type}", ptx_type=ptx_type, mode=mode)
        for mode in ("hi", "lo")
        for ptx_type in ("u32", "s32")
    ),
    *(
        _Form("mad_wide", f"mad.wide.{ptx_type}", ptx_type=ptx_type, mode="wide")
        for ptx_type in _WIDE_TYPES
    ),
    *(
        _Form(
            "dp2a",
            f"dp2a.{mode}.{atype}.{btype}",
            mode=mode,
            atype=atype,
            btype=btype,
        )
        for mode in ("lo", "hi")
        for atype in ("u32", "s32")
        for btype in ("u32", "s32")
    ),
    *(
        _Form("dp4a", f"dp4a.{atype}.{btype}", atype=atype, btype=btype)
        for atype in ("u32", "s32")
        for btype in ("u32", "s32")
    ),
    *(_Form("neg_int", f"neg.{ptx_type}", ptx_type=ptx_type) for ptx_type in ("s16", "s32", "s64")),
)
assert len(_FORMS) == 50

_DTYPES = tuple(_dtype(ptx_type) for ptx_type in _INTEGER_TYPES)
_BUFFER_ROLES = ("a", "b", "c", "output")


def _all_forms_kernel():
    parameters = ",\n".join(
        f'    {role}_{dtype}: T.Buffer(({len(_FORMS)}, 32), "{dtype}")'
        for role in _BUFFER_ROLES
        for dtype in _DTYPES
    )
    calls = []
    for index, form in enumerate(_FORMS):
        operands = [f"output_{form.destination_dtype}[{index}, lane]"]
        operands.extend(
            f"{role}_{dtype}[{index}, lane]"
            for role, dtype in zip(("a", "b", "c"), form.source_dtypes, strict=False)
        )
        calls.append(f'    T.ptx["{form.spelling}"]({", ".join(operands)})')
    calls_text = "\n".join(calls)
    return tvm.script.from_source(
        f"""
@T.prim_func
def kernel(
{parameters}
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
{calls_text}
""",
        {"T": T},
    )


PTX_INTEGER_ARITHMETIC_FORMS = _all_forms_kernel()


def _wrap(value: int, dtype: str) -> int:
    info = np.dtype(dtype)
    bits = info.itemsize * 8
    value &= (1 << bits) - 1
    if info.kind == "i" and value & (1 << (bits - 1)):
        value -= 1 << bits
    return value


def _truncate_div(lhs: int, rhs: int) -> int:
    quotient = abs(lhs) // abs(rhs)
    return -quotient if (lhs < 0) != (rhs < 0) else quotient


def _subword(word: int, index: int, width: int, *, signed: bool) -> int:
    value = ((word & 0xFFFF_FFFF) >> (index * width)) & ((1 << width) - 1)
    if signed and value & (1 << (width - 1)):
        value -= 1 << width
    return value


def _operands(form: _Form, lane: int) -> tuple[int, ...]:
    if form.op == "div":
        bits = int(form.ptx_type[1:])
        if form.ptx_type.startswith("s"):
            lhs = (-(1 << (bits - 1))) if lane == 0 else (lane - 17) * 97
            rhs = 2 if lane % 3 == 0 else (-3 if lane % 3 == 1 else 5)
            return _wrap(lhs, _dtype(form.ptx_type)), _wrap(rhs, _dtype(form.ptx_type))
        return ((1 << (bits - 1)) + lane * 977) & ((1 << bits) - 1), lane % 13 + 1
    if form.op == "rem":
        if form.ptx_type.startswith("s"):
            magnitude = 101 + lane * 11
            divisor = 3 + lane % 7
            sign = -1 if lane % 2 == 0 else 1
            return sign * magnitude, sign * divisor
        bits = int(form.ptx_type[1:])
        return ((1 << (bits - 1)) + lane * 313) & ((1 << bits) - 1), lane % 11 + 1
    if form.op == "sad":
        dtype = _dtype(form.ptx_type)
        bits = int(form.ptx_type[1:])
        if form.ptx_type.startswith("s"):
            low, high = -(1 << (bits - 1)), (1 << (bits - 1)) - 1
            lhs, rhs = (low, high) if lane % 2 == 0 else (high, low + lane)
            return lhs, rhs, _wrap(high - lane * 17, dtype)
        mask = (1 << bits) - 1
        return (lane * 991) & mask, (mask - lane * 313) & mask, (mask - lane) & mask
    if form.op in {"mul24", "mad24"}:
        dtype = _dtype(form.ptx_type)
        lhs = _wrap(0xA500_0000 | ((0x80_0001 + lane * 0x01_0101) & 0xFF_FFFF), dtype)
        rhs = _wrap(0x5A00_0000 | ((0x7F_FFFD - lane * 0x00_1011) & 0xFF_FFFF), dtype)
        if form.op == "mul24":
            return lhs, rhs
        addend = _wrap((1 << 31) - 1 - lane * 101 if form.sat else 0xE100_0003 + lane, dtype)
        return lhs, rhs, addend
    if form.op in {"mul_wide", "mad_wide"}:
        source_dtype = _dtype(form.ptx_type)
        bits = int(form.ptx_type[1:])
        if form.ptx_type.startswith("s"):
            lhs = -(1 << (bits - 1)) + lane
            rhs = (1 << (bits - 1)) - 1 - 3 * lane
        else:
            lhs = ((1 << bits) - 1 - lane) & ((1 << bits) - 1)
            rhs = (0x1235 + lane * 37) & ((1 << bits) - 1)
        if form.op == "mul_wide":
            return _wrap(lhs, source_dtype), _wrap(rhs, source_dtype)
        addend = _wrap((1 << (bits * 2)) - 19 + lane * 7, form.destination_dtype)
        return _wrap(lhs, source_dtype), _wrap(rhs, source_dtype), addend
    if form.op in {"dp2a", "dp4a"}:
        a_word = (0x8001_7FFF ^ (lane * 0x0102_0409)) & 0xFFFF_FFFF
        b_word = (0xFD03_FE04 ^ (lane * 0x0804_0201)) & 0xFFFF_FFFF
        accumulator = _wrap(0x7FFF_FFC0 + lane * 29, form.destination_dtype)
        return (
            _wrap(a_word, _dtype(form.atype)),
            _wrap(b_word, _dtype(form.btype)),
            accumulator,
        )
    dtype = _dtype(form.ptx_type)
    bits = int(form.ptx_type[1:])
    values = (-(1 << (bits - 1)), (1 << (bits - 1)) - 1, -1, 0, 1)
    return (_wrap(values[lane % len(values)] + lane // len(values), dtype),)


def _reference(form: _Form, operands: tuple[int, ...]) -> int:
    if form.op == "div":
        return _wrap(_truncate_div(*operands), form.destination_dtype)
    if form.op == "rem":
        lhs, rhs = operands
        return _wrap(lhs - _truncate_div(lhs, rhs) * rhs, form.destination_dtype)
    if form.op == "sad":
        lhs, rhs, addend = operands
        difference = rhs - lhs if lhs < rhs else lhs - rhs
        return _wrap(addend + difference, form.destination_dtype)
    if form.op in {"mul_wide", "mad_wide"}:
        result = operands[0] * operands[1]
        if form.op == "mad_wide":
            result += operands[2]
        return _wrap(result, form.destination_dtype)
    if form.op in {"mul24", "mad24"}:
        signed = form.ptx_type == "s32"
        product = _subword(operands[0], 0, 24, signed=signed) * _subword(
            operands[1], 0, 24, signed=signed
        )
        result = product >> 16 if form.mode == "hi" else product
        if form.op == "mad24":
            result += operands[2]
            if form.sat:
                return min(max(result, -(1 << 31)), (1 << 31) - 1)
        return _wrap(result, form.destination_dtype)
    if form.op in {"dp2a", "dp4a"}:
        a, b, accumulator = operands
        signed_a = form.atype == "s32"
        signed_b = form.btype == "s32"
        if form.op == "dp4a":
            products = (
                _subword(a, index, 8, signed=signed_a) * _subword(b, index, 8, signed=signed_b)
                for index in range(4)
            )
        else:
            first_byte = 2 if form.mode == "hi" else 0
            products = (
                _subword(a, index, 16, signed=signed_a)
                * _subword(b, first_byte + index, 8, signed=signed_b)
                for index in range(2)
            )
        return _wrap(accumulator + sum(products), form.destination_dtype)
    assert form.op == "neg_int"
    return _wrap(-operands[0], form.destination_dtype)


@pytest.fixture(scope="module")
def integer_arithmetic_results(tmp_path_factory):
    shape = (len(_FORMS), 32)
    arguments = {
        f"{role}_{dtype}": np.zeros(shape, dtype=np.dtype(dtype))
        for role in _BUFFER_ROLES
        for dtype in _DTYPES
    }
    for dtype in _DTYPES:
        arguments[f"output_{dtype}"].fill(_wrap(0xDEAD_BEEF_DEAD_BEEF, dtype))

    expected = {}
    for index, form in enumerate(_FORMS):
        rows = []
        for lane in range(32):
            operands = _operands(form, lane)
            for role, dtype, value in zip(
                ("a", "b", "c")[: len(form.source_dtypes)],
                form.source_dtypes,
                operands,
                strict=True,
            ):
                arguments[f"{role}_{dtype}"][index, lane] = value
            rows.append(_reference(form, operands))
        expected[index] = np.asarray(rows, dtype=np.dtype(form.destination_dtype))

    cache_dir = tmp_path_factory.mktemp("ptx_integer_arithmetic")
    result = numsim.Engine().run(
        numsim.transpile(PTX_INTEGER_ARITHMETIC_FORMS, cache_dir=cache_dir),
        arguments,
        outputs=tuple(f"output_{dtype}" for dtype in _DTYPES),
    )
    return result.outputs, expected


def _assert_operation(results, op: str) -> None:
    outputs, expected = results
    forms = [(index, form) for index, form in enumerate(_FORMS) if form.op == op]
    for index, form in forms:
        np.testing.assert_array_equal(
            outputs[f"output_{form.destination_dtype}"][index],
            expected[index],
            err_msg=form.spelling,
        )


def test_ptx_div_all_legal_forms_match_independent_integer_oracle(integer_arithmetic_results):
    _assert_operation(integer_arithmetic_results, "div")


def test_ptx_rem_all_legal_forms_match_independent_integer_oracle(integer_arithmetic_results):
    _assert_operation(integer_arithmetic_results, "rem")


def test_ptx_sad_all_legal_forms_match_independent_integer_oracle(integer_arithmetic_results):
    _assert_operation(integer_arithmetic_results, "sad")


def test_ptx_mul24_all_legal_forms_match_independent_integer_oracle(integer_arithmetic_results):
    _assert_operation(integer_arithmetic_results, "mul24")


def test_ptx_mul_wide_all_legal_forms_match_independent_integer_oracle(
    integer_arithmetic_results,
):
    _assert_operation(integer_arithmetic_results, "mul_wide")


def test_ptx_mad24_all_legal_forms_match_independent_integer_oracle(integer_arithmetic_results):
    _assert_operation(integer_arithmetic_results, "mad24")


def test_ptx_mad_wide_all_legal_forms_match_independent_integer_oracle(
    integer_arithmetic_results,
):
    _assert_operation(integer_arithmetic_results, "mad_wide")


def test_ptx_dp2a_all_legal_forms_match_independent_packed_lane_oracle(
    integer_arithmetic_results,
):
    _assert_operation(integer_arithmetic_results, "dp2a")


def test_ptx_dp4a_all_legal_forms_match_independent_packed_lane_oracle(
    integer_arithmetic_results,
):
    _assert_operation(integer_arithmetic_results, "dp4a")


def test_ptx_neg_int_all_legal_forms_match_wrapping_negation_oracle(integer_arithmetic_results):
    _assert_operation(integer_arithmetic_results, "neg_int")


@T.prim_func
def ptx_div_s32_error_path(
    lhs: T.Buffer((32,), "int32"),
    rhs: T.Buffer((32,), "int32"),
    output: T.Buffer((32,), "int32"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    T.ptx.div.s32(output[lane], lhs[lane], rhs[lane])


@T.prim_func
def ptx_rem_s32_error_path(
    lhs: T.Buffer((32,), "int32"),
    rhs: T.Buffer((32,), "int32"),
    output: T.Buffer((32,), "int32"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    T.ptx.rem.s32(output[lane], lhs[lane], rhs[lane])


@pytest.mark.parametrize(
    ("kernel", "symbol"),
    ((ptx_div_s32_error_path, "/"), (ptx_rem_s32_error_path, "%")),
    ids=("div", "rem"),
)
def test_ptx_integer_division_by_zero_fails_closed_at_the_faulting_lane(tmp_path, kernel, symbol):
    rhs = np.ones(32, dtype=np.int32)
    rhs[7] = 0
    with pytest.raises(numsim.NumSimExecutionError, match=rf"lane 7: 29 {symbol} 0"):
        numsim.Engine().run(
            numsim.transpile(kernel, cache_dir=tmp_path),
            {
                "lhs": np.full(32, 29, dtype=np.int32),
                "rhs": rhs,
                "output": np.zeros(32, dtype=np.int32),
            },
            outputs=("output",),
        )


@pytest.mark.parametrize(
    ("kernel", "symbol"),
    ((ptx_div_s32_error_path, "/"), (ptx_rem_s32_error_path, "%")),
    ids=("div", "rem"),
)
def test_ptx_signed_division_overflow_fails_closed(tmp_path, kernel, symbol):
    with pytest.raises(
        numsim.NumSimExecutionError,
        match=rf"lane 0: -2147483648 {symbol} -1",
    ):
        numsim.Engine().run(
            numsim.transpile(kernel, cache_dir=tmp_path),
            {
                "lhs": np.full(32, np.iinfo(np.int32).min, dtype=np.int32),
                "rhs": np.full(32, -1, dtype=np.int32),
                "output": np.zeros(32, dtype=np.int32),
            },
            outputs=("output",),
        )


def test_ptx_signed_rem_machine_specific_negative_rounding_fails_closed(tmp_path):
    with pytest.raises(numsim.NumSimExecutionError, match="machine-specific negative operand"):
        numsim.Engine().run(
            numsim.transpile(ptx_rem_s32_error_path, cache_dir=tmp_path),
            {
                "lhs": np.full(32, -5, dtype=np.int32),
                "rhs": np.full(32, 2, dtype=np.int32),
                "output": np.zeros(32, dtype=np.int32),
            },
            outputs=("output",),
        )


__all__ = [
    "PTX_INTEGER_ARITHMETIC_FORMS",
    "ptx_div_s32_error_path",
    "ptx_rem_s32_error_path",
]
