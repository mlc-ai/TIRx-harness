"""Predicate contracts for every comparison entry and both clmad variants."""

import numpy as np
import pytest
import tvm
from tvm.script import tirx as T

from tirx_harness import numsim, racecheck, synccheck


# Spelling, number of destinations, source expressions. Existing numerical
# tests own comparison/GF(2) oracles; this case adds masking and GPU bit parity.
_FORMS = (
    ("set.lt.u32.s32", 1, "i, j"),
    ("set.lt.xor.u32.s32", 1, "i, j, p"),
    ("set.eq.u32.f16x2", 1, "x, y"),
    ("set.ne.and.u32.f16x2", 1, "x, y, p"),
    ("setp.lt.s32", 1, "i, j"),
    ("setp.eq.f16", 1, "h, k"),
    ("setp.lt.and.s32", 1, "i, j, p"),
    ("setp.eq.xor.s32", 2, "i, j, p"),
    ("setp.lt.or.f16", 1, "h, k, p"),
    ("setp.eq.and.f16x2", 2, "x, y, p"),
    ("setp.eq.f16x2", 2, "x, y"),
    ("setp.ne.s32", 2, "i, i"),  # Clears its aliased first output, sets the second.
    ("slct.b32.s32", 1, "x, y, i"),
    ("testp.normal.f32", 1, "f"),
    ("clmad.lo.u64", 1, "a, b, a"),
    ("clmad.hi.u64", 1, "a, b, a"),
)
_ROWS = sum(count for _, count, _ in _FORMS)


def compare_predicate_case(preserve=True):
    statements = []
    row = 0
    for spelling, count, sources in _FORMS:
        dtype = "uint64" if spelling.startswith("clmad") else "uint32"
        slots = "wide" if dtype == "uint64" else "dst"
        for mode in range(2):
            a, b = ("lhs[lane]", "rhs[lane]") if mode == 0 else ("staged[0]", "staged[1]")
            expressions = dict(
                a=a,
                b=b,
                x=f'T.Cast("uint32", {a})',
                y=f'T.Cast("uint32", {b})',
                i=f'T.Cast("int32", {a})',
                j=f'T.Cast("int32", {b})',
                h=f'T.Cast("uint16", {a})',
                k=f'T.Cast("uint16", {b})',
                p=f"({a} & T.uint64(1)) != T.uint64(0)",
                f=f'T.reinterpret("float32", T.Cast("uint32", {a}))',
            )
            alias = mode == 1 and spelling == "setp.ne.s32"
            statements += [f"    {slots}[0] = 91", f"    {slots}[1] = 91"]
            if alias:
                statements.append("    dst[0] = T.Select(active, T.uint32(91), T.uint32(0))")
            arguments = [f"{slots}[{index}]" for index in range(count)]
            arguments += [expressions[source] for source in sources.split(", ")]
            if mode:
                arguments += [
                    f"pred={'dst[0] != 0' if alias else 'active'}",
                    f"preserve_dst={preserve}",
                ]
            statements.append(f'    T.ptx["{spelling}"]({", ".join(arguments)})')
            statements += [
                f'    output[{mode}, {row + index}, lane] = T.Cast("uint64", {slots}[{index}])'
                for index in range(count)
            ]
        row += count
    return tvm.script.from_source(
        f"""
@T.prim_func
def kernel(lhs: T.Buffer((32,), "uint64"), rhs: T.Buffer((32,), "uint64"),
           output: T.Buffer((2, {_ROWS}, 32), "uint64"), selected: T.uint32):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    active = (selected & (T.uint32(1) << T.Cast("uint32", lane))) != T.uint32(0)
    staged = T.alloc_local((2,), "uint64")
    dst = T.alloc_local((2,), "uint32")
    wide = T.alloc_local((2,), "uint64")
    if active:
        staged[0] = lhs[lane]
        staged[1] = rhs[lane]
{chr(10).join(statements)}
""",
        {"T": T},
    )


def compare_predicate_inputs(mask):
    # Ordinary values, sign bits, packed-half subnormals/infinities/NaNs, and
    # independent high halves for clmad. Registers, not GPU memory, are left
    # uninitialized on non-participating lanes.
    bits = np.resize(
        np.array(
            [
                0,
                0x80000000,
                0x00010001,
                0x80018001,
                0x3C003C00,
                0xBC003C00,
                0x7C00FC00,
                0x7E017E02,
                0x7F800000,
                0x7FC00001,
                0xFFFFFFFF,
                17,
            ],
            np.uint64,
        ),
        32,
    )
    lhs = bits | (np.arange(32, dtype=np.uint64) * np.uint64(0x8000000100000000))
    return {
        "lhs": lhs,
        "rhs": np.roll(lhs, 3),
        "selected": mask,
        "output": np.zeros((2, _ROWS, 32), np.uint64),
    }


def check_compare_predicates(inputs, output, preserve):
    active = np.array([bool(inputs["selected"] & (1 << lane)) for lane in range(32)])
    row = 0
    for spelling, count, _ in _FORMS:
        baseline, masked = output[:, row : row + count]
        np.testing.assert_array_equal(masked[:, active], baseline[:, active], err_msg=spelling)
        kept = int(bool(91)) if spelling.startswith(("setp", "testp")) else 91
        expected = np.full((count, int((~active).sum())), kept if preserve else 0, np.uint64)
        if spelling == "setp.ne.s32":
            expected[0] = 0
            np.testing.assert_array_equal(masked[1, active], 1)
        np.testing.assert_array_equal(masked[:, ~active], expected, err_msg=spelling)
        row += count


@pytest.mark.parametrize("preserve", [False, True])
def test_compare_instruction_predicates(preserve, tmp_path):
    kernel = compare_predicate_case(preserve)
    module = numsim.transpile(kernel, cache_dir=tmp_path)
    for mask in (0, 0x80000001, 0xAAAAAAAA, 0xFFFFFFFF):
        inputs = compare_predicate_inputs(mask)
        for checker in (synccheck, racecheck):
            checker(kernel, inputs).require_clean()
        actual = numsim.Engine().run(module, inputs).outputs["output"]
        check_compare_predicates(inputs, actual, preserve)
