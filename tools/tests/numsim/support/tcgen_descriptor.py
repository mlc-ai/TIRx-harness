"""tcgen05 instruction-descriptor encoders (the oracle for expected descriptor words).

Copied verbatim from the deleted Python frontend (`ops/tcgen_descriptor.py`); the
native frontend owns the production encoding in `analyze/tcgen_descriptor.rs`.
"""

from __future__ import annotations

from tirx_harness.numsim.errors import UnsupportedTIRxError

MATRIX_DESC = "tirx.cuda.tcgen05_encode_matrix_descriptor"
INSTR_DESC = "tirx.cuda.tcgen05_encode_instr_descriptor"
INSTR_DESC_BLOCK = "tirx.cuda.tcgen05_encode_instr_descriptor_block_scaled"

_FORMAT_MAP = {
    "float16": 0,
    "bfloat16": 1,
    "tf32": 2,
    "float8_e4m3fn": 0,
    "float8_e4m3fnuz": 0,
    "float8_e5m2": 1,
    "float6_e2m3fn": 3,
    "float6_e3m2fn": 4,
    "float4_e2m1fn": 5,
    "uint8": 0,
    "int8": 1,
    "float32": 1,
    "int32": 2,
}
_FP8_FAMILY = frozenset(
    {
        "float8_e4m3fn",
        "float8_e4m3fnuz",
        "float8_e5m2",
        "float6_e2m3fn",
        "float6_e3m2fn",
        "float4_e2m1fn",
    }
)
_EIGHT_BIT_SOURCE_DTYPES = frozenset(
    {"float8_e4m3fn", "float8_e4m3fnuz", "float8_e5m2", "int8", "uint8"}
)


def _dense_kind(d_dtype: str, a_dtype: str, b_dtype: str) -> str:
    # PTX ISA 9.7.17.4.2 Table 45: under `.kind::f16` the D format is F16 or
    # F32 and each multiplicand is independently F16 or BF16.
    if (
        d_dtype in {"float16", "float32"}
        and a_dtype in {"float16", "bfloat16"}
        and b_dtype in {"float16", "bfloat16"}
    ):
        return "f16"
    if d_dtype == "float32" and a_dtype == b_dtype == "tf32":
        return "tf32"
    if (
        d_dtype == "int32"
        and a_dtype in {"int8", "uint8"}
        and b_dtype
        in {
            "int8",
            "uint8",
        }
    ):
        return "i8"
    if d_dtype in {"float16", "float32"} and a_dtype in _FP8_FAMILY and b_dtype in _FP8_FAMILY:
        return "f8f6f4"
    raise UnsupportedTIRxError(
        "tcgen05 dense instruction descriptor has invalid dtype combination "
        f"D={d_dtype}, A={a_dtype}, B={b_dtype}"
    )


def _block_kind(d_dtype: str, a_dtype: str, b_dtype: str, sfa_dtype: str, sfb_dtype: str) -> str:
    if d_dtype != "float32":
        raise UnsupportedTIRxError("tcgen05 block-scaled instruction descriptor requires float32 D")
    if a_dtype == b_dtype == "float4_e2m1fn" and sfa_dtype == sfb_dtype == "float8_e4m3fn":
        return "mxf4nvf4"
    if a_dtype == b_dtype == "float4_e2m1fn" and sfa_dtype == sfb_dtype == "float8_e8m0fnu":
        return "mxf4"
    if (
        a_dtype in _FP8_FAMILY
        and b_dtype in _FP8_FAMILY
        and sfa_dtype == sfb_dtype == "float8_e8m0fnu"
    ):
        return "mxf8f6f4"
    raise UnsupportedTIRxError(
        "tcgen05 block-scaled instruction descriptor has invalid dtype combination "
        f"D={d_dtype}, A={a_dtype}, B={b_dtype}, SFA={sfa_dtype}, SFB={sfb_dtype}"
    )


def validate_tcgen05_instruction_shape(
    kind: str, cta_group: int, m: int, n: int, k: int, sparse: bool = False
) -> None:
    if cta_group not in {1, 2}:
        raise UnsupportedTIRxError(
            f"tcgen05 instruction descriptor cta_group must be 1 or 2, got {cta_group}"
        )
    if kind in {"f16", "tf32", "f8f6f4"}:
        steps = {64: 8, 128: 8} if cta_group == 1 else {128: 16, 256: 16}
        expected_k = {"f16": 16, "tf32": 8, "f8f6f4": 32}[kind]
        extras: set[int] = set()
    elif kind == "i8":
        steps = {64: 16, 128: 16} if cta_group == 1 else {128: 32, 256: 32}
        expected_k = 32
        extras = {8, 24} if cta_group == 1 else set()
    elif kind in {"mxf8f6f4", "mxf4", "mxf4nvf4"}:
        steps = {128: 8} if cta_group == 1 else {128: 16, 256: 16}
        expected_k = 32 if kind == "mxf8f6f4" else 64
        extras = set()
    else:
        raise UnsupportedTIRxError(f"unknown tcgen05 instruction descriptor kind {kind!r}")
    if sparse:
        expected_k *= 2
        if kind in {"mxf8f6f4", "mxf4", "mxf4nvf4"} and cta_group == 2 and m != 256:
            raise UnsupportedTIRxError(
                f"invalid sparse tcgen05 block-scaled descriptor shape kind={kind}, "
                f"cta_group={cta_group}, M={m}, N={n}, K={k}; CTA group 2 requires M=256"
            )
    step = steps.get(m)
    if (
        step is None
        or (n not in extras and not (step <= n <= 256 and n % step == 0))
        or k != expected_k
    ):
        raise UnsupportedTIRxError(
            f"invalid tcgen05 descriptor shape kind={kind}, cta_group={cta_group}, M={m}, N={n}, K={k}"
        )


def _validate_8bit_transpose_b_shape(
    *, b_dtype: str, trans_b: bool, cta_group: int, n: int
) -> None:
    if not trans_b or b_dtype not in _EIGHT_BIT_SOURCE_DTYPES:
        return
    step = 16 if cta_group == 1 else 32
    if n < step or n > 256 or n % step != 0:
        raise UnsupportedTIRxError(
            "tcgen05 8-bit transpose B requires "
            f"cta_group={cta_group} N in [{step}, 256] with step {step}, got N={n}"
        )


def encode_dense_instr_descriptor_fields(
    *,
    d_dtype: str,
    a_dtype: str,
    b_dtype: str,
    m: int,
    n: int,
    k: int,
    trans_a: bool,
    trans_b: bool,
    cta_group: int,
    neg_a: bool = False,
    neg_b: bool = False,
    sat_d: bool = False,
    sparse: bool = False,
) -> int:
    kind = _dense_kind(d_dtype, a_dtype, b_dtype)
    validate_tcgen05_instruction_shape(kind, cta_group, m, n, k, sparse)
    transpose_dtypes = {
        "float8_e4m3fn",
        "float8_e4m3fnuz",
        "float8_e5m2",
        "int8",
        "uint8",
        "float16",
        "bfloat16",
        "tf32",
    }
    if trans_a and a_dtype not in transpose_dtypes:
        raise UnsupportedTIRxError(f"tcgen05 transpose A is invalid for {a_dtype}")
    if trans_b and b_dtype not in transpose_dtypes:
        raise UnsupportedTIRxError(f"tcgen05 transpose B is invalid for {b_dtype}")
    _validate_8bit_transpose_b_shape(b_dtype=b_dtype, trans_b=trans_b, cta_group=cta_group, n=n)
    if (neg_a or neg_b) and kind not in {"f16", "tf32", "f8f6f4"}:
        raise UnsupportedTIRxError(f"tcgen05 negate is invalid for kind {kind}")
    if sat_d and kind != "i8":
        raise UnsupportedTIRxError(f"tcgen05 saturation is invalid for kind {kind}")

    value = int(sparse) << 2
    value |= int(sat_d) << 3
    value |= (_FORMAT_MAP[d_dtype] & 0x3) << 4
    value |= (_FORMAT_MAP[a_dtype] & 0x7) << 7
    value |= (_FORMAT_MAP[b_dtype] & 0x7) << 10
    value |= int(neg_a) << 13
    value |= int(neg_b) << 14
    value |= int(trans_a) << 15
    value |= int(trans_b) << 16
    value |= ((n >> 3) & 0x3F) << 17
    value |= ((m >> 4) & 0x1F) << 24
    return value & 0xFFFFFFFF


def encode_block_scaled_instr_descriptor_fields(
    *,
    d_dtype: str,
    a_dtype: str,
    b_dtype: str,
    sfa_dtype: str,
    sfb_dtype: str,
    m: int,
    n: int,
    k: int,
    trans_a: bool,
    trans_b: bool,
    cta_group: int,
    neg_a: bool = False,
    neg_b: bool = False,
    sparse: bool = False,
) -> int:
    kind = _block_kind(d_dtype, a_dtype, b_dtype, sfa_dtype, sfb_dtype)
    validate_tcgen05_instruction_shape(kind, cta_group, m, n, k, sparse)
    fp8 = {"float8_e4m3fn", "float8_e4m3fnuz", "float8_e5m2"}
    if trans_a and a_dtype not in fp8:
        raise UnsupportedTIRxError(f"tcgen05 block transpose A is invalid for {a_dtype}")
    if trans_b and b_dtype not in fp8:
        raise UnsupportedTIRxError(f"tcgen05 block transpose B is invalid for {b_dtype}")
    _validate_8bit_transpose_b_shape(b_dtype=b_dtype, trans_b=trans_b, cta_group=cta_group, n=n)

    a_format = _FORMAT_MAP[a_dtype] if kind == "mxf8f6f4" else 1
    b_format = _FORMAT_MAP[b_dtype] if kind == "mxf8f6f4" else 1
    scale_format = int(sfa_dtype == "float8_e8m0fnu")
    value = int(sparse) << 2
    value |= (a_format & 0x7) << 7
    value |= (b_format & 0x7) << 10
    value |= int(neg_a) << 13
    value |= int(neg_b) << 14
    value |= int(trans_a) << 15
    value |= int(trans_b) << 16
    value |= ((n >> 3) & 0x3F) << 17
    value |= (scale_format & 0x1) << 23
    value |= ((m >> 4) & 0x1F) << 24
    return value & 0xFFFFFFFF
