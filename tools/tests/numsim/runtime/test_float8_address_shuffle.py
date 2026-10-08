from __future__ import annotations

import numpy as np
import pytest
import tvm

from tirx_harness import numsim
from tvm.script import tirx as T


@T.prim_func
def float8_address(source: T.Buffer((32,), "float8_e4m3fn")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    T.evaluate(T.address_of(source[lane]))


@T.prim_func
def cuda_shfl_sync_u32(source: T.Buffer((32,), "uint32"), output: T.Buffer((32,), "uint32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    output[lane] = T.cuda.__shfl_sync(
        T.uint32(0xFFFFFFFF), source[lane], T.cast(31 - lane, "uint32"), 32
    )


def _fp8_identity_reinterpret_roundtrip(dtype: str):
    return tvm.script.from_source(
        f'''\
@T.prim_func
def fp8_identity_reinterpret_roundtrip(
    source: T.Buffer((256,), "{dtype}"),
    output: T.Buffer((256,), "{dtype}"),
):
    T.device_entry()
    warp = T.warp_id([8])
    lane = T.lane_id([32])
    index = warp * 32 + lane
    output[index] = T.call_intrin("{dtype}", "tirx.reinterpret", source[index])
''',
        extra_vars={"T": T},
    )


def test_float8_address_uses_one_byte_physical_pointer(tmp_path):
    source = np.zeros(32, dtype=np.uint8)

    module = numsim.transpile(float8_address, cache_dir=tmp_path)
    result = numsim.Engine().run(module, {"source": source})

    assert set(result.outputs) == {"source"}
    np.testing.assert_array_equal(result.outputs["source"], 0)
    assert "PhysicalPtr::new" in module.rust_source
    assert "tirx.address_of" not in module.rust_source


@pytest.mark.parametrize("dtype", ["float8_e4m3fn", "float8_e8m0fnu"])
def test_scalar_fp8_identity_reinterpret_rejects_256_payload_roundtrip(dtype, tmp_path):
    with pytest.raises(
        numsim.UnsupportedTIRxError,
        match="raw payload reinterpret is not modeled for scalar low-precision",
    ):
        numsim.transpile(_fp8_identity_reinterpret_roundtrip(dtype), cache_dir=tmp_path)


def test_cuda_four_argument_shfl_sync_uses_implicit_warp_size(tmp_path):
    source = np.arange(32, dtype=np.uint32) * np.uint32(7) + np.uint32(3)
    output = np.zeros(32, dtype=np.uint32)

    module = numsim.transpile(cuda_shfl_sync_u32, cache_dir=tmp_path)
    result = numsim.Engine().run(module, {"source": source, "output": output})

    np.testing.assert_array_equal(result.outputs["output"], source[::-1])
    assert "fn warp_shuffle(" not in module.rust_source
    assert "tirx.cuda.__shfl_sync" not in module.rust_source
