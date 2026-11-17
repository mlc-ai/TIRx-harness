"""sm_107f/sm_120f packed integer add/sub/neg/min/max (PTX ISA 9.2 forms).

The oracle splits each `.b32` carrier into independent 8- or 16-bit lanes with
numpy views and evaluates every lane in int64, so wrap-around, saturation and
`.relu` follow from PTX 9.7.1.1/2/12/13/14 rather than from engine helpers.
"""

import numpy as np

from tests.numsim.support.execution import run_checked
from tests.numsim.support.register_cases import masked_arithmetic_case

SIZE = 128

# Every legal spelling of add_int_sm107f, sub_int_sm107f, neg_int_sm107f,
# min_int8x4 and max_int8x4.
BINARY_FORMS = (
    "add.u8x4",
    "add.s8x4",
    "add.sat.u8x4",
    "add.sat.s8x4",
    "add.sat.u32",
    "add.sat.u16x2",
    "add.sat.s16x2",
    "sub.u8x4",
    "sub.s8x4",
    "sub.sat.u8x4",
    "sub.sat.s8x4",
    "min.u8x4",
    "min.s8x4",
    "min.relu.s8x4",
    "max.u8x4",
    "max.s8x4",
    "max.relu.s8x4",
)
UNARY_FORMS = ("neg.s8x4",)

# Lane-boundary pairs; random words fill the rest of the launch.
EDGE_PAIRS = (
    (0x7F7F7F7F, 0x01010101),  # s8 MAXINT + 1
    (0x80808080, 0x01010101),  # s8 MININT - 1 (sub), -128 negation
    (0xFFFFFFFF, 0x01010101),  # u8 0xff + 1, s8 -1 + 1, u32 MAXINT + 0x01010101
    (0x00000000, 0x01010101),  # u8 0 - 1
    (0x807FFF00, 0x01FF0180),  # one edge per lane, mixed directions
    (0x81FF7F80, 0x7F0180FF),
    (0x7FFF8000, 0x00010001),  # s16 MAXINT + 1 / u16 0x8000 + 1
    (0x80008000, 0xFFFFFFFF),  # s16 MININT + -1
    (0xFFFF0001, 0x0001FFFF),  # u16 overflow in each half separately
    (0xFFFFFFFE, 0x00000005),  # u32 saturation
    (0x80000000, 0x80000000),  # u32 wrap avoided by .sat; s8/s16 MININT + MININT
    (0x7F80FF01, 0x7F80FF01),  # equal operands
    (0x80FF0102, 0xFF800201),  # relu sees negative minimum and maximum lanes
)


def _inputs():
    rng = np.random.default_rng(107)
    a = rng.integers(0, 1 << 32, SIZE, dtype=np.uint64).astype(np.uint32)
    b = rng.integers(0, 1 << 32, SIZE, dtype=np.uint64).astype(np.uint32)
    for row, (lhs, rhs) in enumerate(EDGE_PAIRS):
        # Even lanes run unmasked in both halves; place each pair on one of each parity.
        a[2 * row], b[2 * row] = lhs, rhs
        a[2 * row + 1], b[2 * row + 1] = rhs, lhs
    return a, b


def _lanes(words, lane_dtype):
    return words.astype("<u4").view(lane_dtype).astype(np.int64)


def _pack(values, lane_dtype):
    info = np.iinfo(lane_dtype)
    wrapped = (values - info.min) % (1 << info.bits) + info.min
    return wrapped.astype(lane_dtype).view("<u4").astype(np.uint32)


def _reference(spelling, a, b):
    mnemonic, *modifiers, ptx_type = spelling.split(".")
    lane_dtype = {
        "u8x4": np.uint8,
        "s8x4": np.int8,
        "u16x2": np.uint16,
        "s16x2": np.int16,
        "u32": np.uint32,
    }[ptx_type]
    lhs, rhs = _lanes(a, lane_dtype), _lanes(b, lane_dtype)
    value = {
        "add": lambda: lhs + rhs,
        "sub": lambda: lhs - rhs,
        "neg": lambda: -lhs,
        "min": lambda: np.minimum(lhs, rhs),
        "max": lambda: np.maximum(lhs, rhs),
    }[mnemonic]()
    if "sat" in modifiers:
        info = np.iinfo(lane_dtype)
        value = np.clip(value, info.min, info.max)
    if "relu" in modifiers:
        value = np.maximum(value, 0)
    return _pack(value, lane_dtype)


def sm107_packed_int_case():
    a, b = _inputs()
    forms = [(spelling, ("a", "b")) for spelling in BINARY_FORMS]
    forms += [(spelling, ("a",)) for spelling in UNARY_FORMS]
    preserved = np.full(SIZE, 0x5A5A5A5A, np.uint32)
    kernel, inputs = masked_arithmetic_case(
        forms, {"preserved": preserved, "a": a, "b": b}, preserved="preserved"
    )
    expected = np.stack([_reference(spelling, a, b) for spelling, _ in forms])
    masked = expected.copy()
    masked[:, 1::2] = preserved[1::2]
    return kernel, inputs, np.concatenate([expected, masked]), [s for s, _ in forms]


def test_reference_matches_hand_computed_lane_edges():
    a = np.array([0x807FFF00], np.uint32)
    b = np.array([0x01FF0180], np.uint32)
    # Lanes (low to high) a = 00 ff 7f 80, b = 80 01 ff 01.
    assert _reference("add.u8x4", a, b)[0] == 0x817E0080
    assert _reference("add.sat.u8x4", a, b)[0] == 0x81FFFF80
    assert _reference("add.sat.s8x4", a, b)[0] == 0x817E0080
    assert _reference("sub.sat.u8x4", a, b)[0] == 0x7F00FE00
    assert _reference("sub.sat.s8x4", a, b)[0] == 0x807FFE7F
    assert _reference("neg.s8x4", a, b)[0] == 0x80810100
    assert _reference("min.relu.s8x4", a, b)[0] == 0x00000000
    assert _reference("max.relu.s8x4", a, b)[0] == 0x017F0100
    assert _reference("min.u8x4", a, b)[0] == 0x017F0100
    assert _reference("add.sat.s16x2", np.array([0x7FFF8000]), np.array([0x0001FFFF]))[0] == (
        0x7FFF8000
    )
    assert _reference("add.sat.u32", np.array([0xFFFFFFFE]), np.array([5]))[0] == 0xFFFFFFFF


def test_sm107_packed_integer_forms_match_lane_oracle(tmp_path):
    kernel, inputs, expected, spellings = sm107_packed_int_case()
    result = run_checked(kernel, inputs, cache_dir=tmp_path)
    actual = result.outputs["output"]
    for row, spelling in enumerate(spellings * 2):
        np.testing.assert_array_equal(
            actual[row], expected[row], err_msg=f"row {row}: {spelling}", strict=True
        )
