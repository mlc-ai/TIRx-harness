from __future__ import annotations

import numpy as np
import pytest

from tirx_harness import numsim
from tirx_harness.numsim.errors import UnsupportedTIRxError
from tirx_harness.numsim.transpiler.frontend import analyze, verify
from tvm.script import tirx as T


_SMEM_DESC_ADD_16B_OFFSET_SOURCE = r"""
__forceinline__ __device__ uint64_t tvm_builtin_smem_desc_add_16B_offset(
    uint64_t desc_base, int32_t offset) {
    SmemDescriptor desc;
    desc.desc_ = desc_base;
    desc.lo += static_cast<uint32_t>(offset);
    return desc.desc_;
}
"""


_SMEM_DESC_MAKE_LO_UNIFORM_SOURCE = r"""
__forceinline__ __device__ void smem_desc_make_lo_uniform(uint64_t* desc) {
    SmemDescriptor* d = reinterpret_cast<SmemDescriptor*>(desc);
    d->lo = __shfl_sync(0xffffffff, d->lo, 0);
}
"""


@T.prim_func
def opaque_value_helper(output: T.Buffer((32,), "float32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    output[lane] = T.cuda.func_call(
        "opaque_fdividef",
        T.float32(1),
        T.cast(lane + 1, "float32"),
        source_code="float opaque_fdividef(float, float);",
        return_type="float32",
    )


@T.prim_func
def opaque_statement_helper():
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    if lane == 0:
        T.evaluate(
            T.cuda.func_call(
                "tvm_builtin_tcgen05_mma_mxf4_block32_ss",
                source_code="void tvm_builtin_tcgen05_mma_mxf4_block32_ss();",
                return_type="void",
            )
        )


@T.prim_func
def smem_descriptor_offset_helper(
    descriptors: T.Buffer((32,), "uint64"),
    offsets: T.Buffer((32,), "int32"),
    output: T.Buffer((32,), "uint64"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    output[lane] = T.cuda.func_call(
        "tvm_builtin_smem_desc_add_16B_offset",
        descriptors[lane],
        offsets[lane],
        source_code=_SMEM_DESC_ADD_16B_OFFSET_SOURCE,
        return_type="uint64",
    )


@T.prim_func
def smem_descriptor_make_lo_uniform_helper(
    descriptors: T.Buffer((32,), "uint64"),
    output: T.Buffer((32,), "uint64"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    descriptor = T.alloc_local((1,), "uint64")
    descriptor[0] = descriptors[lane]
    T.evaluate(
        T.cuda.func_call(
            "smem_desc_make_lo_uniform",
            T.address_of(descriptor[0]),
            source_code=_SMEM_DESC_MAKE_LO_UNIFORM_SOURCE,
            return_type="void",
        )
    )
    output[lane] = descriptor[0]


@pytest.mark.parametrize(
    "kernel",
    [
        opaque_value_helper,
        opaque_statement_helper,
    ],
)
def test_opaque_or_spoofed_cuda_helpers_are_rejected(kernel, tmp_path):
    spec = analyze(kernel)

    unsupported = [item for item in spec.unsupported if "tirx.cuda.func_call" in item]
    assert len(unsupported) == 1

    with pytest.raises(UnsupportedTIRxError):
        verify(spec)
    with pytest.raises(UnsupportedTIRxError):
        numsim.transpile(kernel, cache_dir=tmp_path)


def test_smem_descriptor_offset_wraps_only_the_low_32_bits(tmp_path):
    lanes = np.arange(32, dtype=np.uint64)
    descriptors = np.uint64(0xFEDCBA9800000000) | (
        (lanes * np.uint64(0x1020304) + np.uint64(0xFFFF_FFF0)) & np.uint64(0xFFFF_FFFF)
    )
    offsets = (np.arange(32, dtype=np.int64) * 19 - 41).astype(np.int32)

    module = numsim.transpile(smem_descriptor_offset_helper, cache_dir=tmp_path)
    result = numsim.Engine().run(
        module,
        {
            "descriptors": descriptors,
            "offsets": offsets,
            "output": np.zeros(32, dtype=np.uint64),
        },
    )

    expected_low = (
        (descriptors & np.uint64(0xFFFF_FFFF)) + offsets.astype(np.uint32).astype(np.uint64)
    ) & np.uint64(0xFFFF_FFFF)
    expected = (descriptors & np.uint64(0xFFFF_FFFF_0000_0000)) | expected_low
    np.testing.assert_array_equal(result.outputs["output"], expected)


def test_smem_descriptor_make_lo_uniform_broadcasts_only_lane_zero_low_bits(tmp_path):
    lanes = np.arange(32, dtype=np.uint64)
    descriptors = ((np.uint64(0x8000_0000) + lanes * np.uint64(0x0102_0304)) << np.uint64(32)) | (
        np.uint64(0xFEDC_BA98) - lanes * np.uint64(0x1111_1111)
    )

    module = numsim.transpile(smem_descriptor_make_lo_uniform_helper, cache_dir=tmp_path)
    result = numsim.Engine().run(
        module,
        {
            "descriptors": descriptors,
            "output": np.zeros(32, dtype=np.uint64),
        },
    )

    expected = (descriptors & np.uint64(0xFFFF_FFFF_0000_0000)) | (
        descriptors[0] & np.uint64(0xFFFF_FFFF)
    )
    np.testing.assert_array_equal(result.outputs["output"], expected)
