from __future__ import annotations

import numpy as np

from tirx_harness import numsim
from tests.numsim.support.kernels import float8_buffer_codecs


def _decode_e4m3(bits: np.ndarray) -> np.ndarray:
    bits = np.asarray(bits, dtype=np.uint8)
    exponent = ((bits >> np.uint8(3)) & np.uint8(0xF)).astype(np.int16)
    mantissa = (bits & np.uint8(0x7)).astype(np.float32)
    normal = np.ldexp(np.float32(1) + mantissa / np.float32(8), exponent - 7)
    subnormal = np.ldexp(mantissa / np.float32(8), -6)
    result = np.where(exponent == 0, subnormal, normal).astype(np.float32)
    result = np.where((bits & np.uint8(0x80)) != 0, -result, result)
    return np.where((exponent == 15) & ((bits & 7) == 7), np.nan, result).astype(np.float32)


def test_float8_buffers_use_one_byte_physical_storage_and_native_codecs(tmp_path):
    e4m3 = np.tile(np.array([0x00, 0x01, 0x08, 0x38, 0x3C, 0x7E, 0x80, 0xB8], dtype=np.uint8), 4)
    e8m0 = np.arange(112, 144, dtype=np.uint8)
    output_e4m3 = np.zeros(32, dtype=np.uint8)
    decoded = np.zeros((32, 2), dtype=np.float32)

    module = numsim.transpile(float8_buffer_codecs, cache_dir=tmp_path)
    result = numsim.Engine().run(
        module,
        {
            "input_e4m3": e4m3,
            "input_e8m0": e8m0,
            "output_e4m3": output_e4m3,
            "decoded": decoded,
        },
    )

    np.testing.assert_array_equal(result.outputs["output_e4m3"], e4m3)
    np.testing.assert_allclose(result.outputs["decoded"][:, 0], _decode_e4m3(e4m3))
    np.testing.assert_allclose(
        result.outputs["decoded"][:, 1], np.exp2(e8m0.astype(np.int16) - 127).astype(np.float32)
    )
