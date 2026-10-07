from __future__ import annotations

import pytest

from tirx_harness.numsim.dtype_abi import dtype_itemsize, vector_dtype_abi
from tirx_harness.numsim.errors import UnsupportedTIRxError
from tests.numsim.support.manifest import (
    call_op_names,
    evaluated_kernel,
    resolved_kernel,
)
from tvm.backend.cuda.op import cuda_atomic_cas
from tvm.tirx import Cast, Var, decl_buffer


def test_storage_width_does_not_grant_scalar_cast_support():
    source = Var("source", "float32")
    for dtype in ("float8_e3m4", "float8_e4m3fn"):
        assert dtype_itemsize(dtype) == 1
    with pytest.raises(UnsupportedTIRxError, match=r"Cast\(float32->float8_e3m4\)"):
        resolved_kernel(evaluated_kernel(Cast("float8_e3m4", source), (source,)))
    accepted = resolved_kernel(evaluated_kernel(Cast("float8_e4m3fn", source), (source,)))
    assert not accepted.unsupported


@pytest.mark.parametrize(
    ("dtype", "element_dtype", "lanes", "total_bits"),
    [
        ("float16x2", "float16", 2, 32),
        ("bfloat16x2", "bfloat16", 2, 32),
        ("float32x2", "float32", 2, 64),
        ("uint64x2", "uint64", 2, 128),
        ("int8x16", "int8", 16, 128),
        ("float32x4", "float32", 4, 128),
    ],
)
def test_vector_dtype_abi_is_derived_from_element_and_total_width(
    dtype, element_dtype, lanes, total_bits
):
    abi = vector_dtype_abi(dtype)
    assert abi is not None
    assert abi.element_dtype == element_dtype
    assert abi.lanes == lanes
    assert abi.total_bits == total_bits
    assert abi.itemsize == total_bits // 8
    assert dtype_itemsize(dtype) == total_bits // 8


@pytest.mark.parametrize("dtype", ["float32", "float32x1", "float32x3", "unknownx2"])
def test_vector_dtype_abi_rejects_nonvector_or_unrepresentable_widths(dtype):
    assert vector_dtype_abi(dtype) is None


@pytest.mark.parametrize(
    "dtype",
    [
        "int8x16",
        "uint8x16",
        "float8_e3m4x16",
        "float8_e4m3x16",
        "float8_e4m3b11fnuzx16",
        "float8_e4m3fnx16",
        "float8_e4m3fnuzx16",
        "float8_e5m2x16",
        "float8_e5m2fnuzx16",
        "float8_e8m0fnux16",
        "int16x8",
        "uint16x8",
        "float16x8",
        "bfloat16x8",
        "int32x4",
        "uint32x4",
        "float32x4",
        "int64x2",
        "uint64x2",
        "float64x2",
    ],
)
def test_cuda_atomic_cas_emits_every_byte_addressable_128bit_vector(dtype):
    pointer, compare, value = (
        decl_buffer((1,), dtype, name=name) for name in ("memory", "compare", "value")
    )
    call = cuda_atomic_cas(pointer.data, compare.vload([0]), value.vload([0]))
    kernel = resolved_kernel(evaluated_kernel(call, (pointer, compare, value)))
    assert "tirx.cuda.atomic_cas" in call_op_names(kernel)
    assert not kernel.unsupported


@pytest.mark.parametrize(
    "dtype", ["uint32x2", "float32x2", "boolx4", "boolx128", "float4_e2m1fnx32"]
)
def test_cuda_atomic_cas_classifier_rejects_non128_or_non_byte_addressable_vectors(dtype):
    call = cuda_atomic_cas(Var("pointer", "handle"), Var("compare", dtype), Var("value", dtype))
    with pytest.raises(UnsupportedTIRxError, match="128-bit vector operands"):
        resolved_kernel(evaluated_kernel(call, call.args))
