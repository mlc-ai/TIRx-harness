"""Logic/selection operate on payloads, not the numeric carrier interpretation."""

import ml_dtypes
import numpy as np
import pytest
from tvm.script import tirx as T

from tests.numsim.microtests.harness import PairedBuffer
from tests.numsim.support.execution import assert_rejected, run_checked
from tests.numsim.support.register_cases import masked_arithmetic_case


def logic_carrier_case(width, operand=None):
    # Canonical carriers and every logic spelling already have numerical and
    # predicate coverage in test_ptx_bitops. Here exercise floating bit carriers,
    # including NaN payloads, through the same transport path.
    raw = f"uint{width}"
    mask = (1 << width) - 1
    patterns = [
        0,
        1,
        mask,
        1 << (width - 1),
        0x7DFF,
        0x7F81,
        0x7F800001,
        0xFFC01234,
        0x7FF0000000000001,
    ]
    a = np.resize(np.array([v & mask for v in patterns], raw), 32)
    b = np.roll(a, 3)
    forms = [(f"{op}.b{width}", ("a", "b")) for op in ("and", "or", "xor")]
    forms += [(f"{op}.b{width}", ("a",)) for op in ("not", "cnot")]
    expected = np.stack((a & b, a | b, a ^ b, ~a, (a == 0).astype(raw)))
    operand = operand or f"float{width}"
    inputs = {"a": a, "b": b, "preserved": np.full(32, 0x55, raw)}
    masked = expected.copy()
    masked[:, 1::2] = inputs["preserved"][1::2]
    kernel, inputs = masked_arithmetic_case(
        forms, inputs, preserved="preserved", operand_dtype=operand
    )
    return kernel, inputs, np.concatenate((expected, masked))


@pytest.mark.parametrize("width", (16, 32, 64))
def test_logic_carriers_preserve_bits_and_masks(width, tmp_path):
    for operand in ("float16", "bfloat16") if width == 16 else (f"float{width}",):
        kernel, inputs, expected = logic_carrier_case(width, operand)
        result = run_checked(kernel, inputs, cache_dir=tmp_path)
        np.testing.assert_array_equal(result.outputs["output"], expected)


@pytest.mark.numsim_gpu
def test_half_reinterpret_preserves_storage_payloads(gpu_runner, tmp_path):
    for dtype in ("float16", "bfloat16"):

        @T.prim_func
        def kernel(
            raw: T.Buffer((32,), "uint16"),
            typed: T.Buffer((32,), dtype),
            output: T.Buffer((4, 32), "uint16"),
        ):
            T.device_entry()
            _warp = T.warp_id([1])
            lane = T.lane_id([32])
            value = T.alloc_local((1,), dtype)
            bits = value.view("uint16")
            T.ptx.xor.b16(value[0], typed[lane], T.cast(0, dtype))
            output[0, lane] = bits[0]
            output[1, lane] = T.reinterpret("uint16", value[0])
            T.ptx.xor.b16(value[0], T.reinterpret(dtype, raw[lane]), T.cast(0, dtype))
            output[2, lane] = bits[0]
            output[3, lane] = T.reinterpret("uint16", value[0])

        # Both formats' signaling NaNs, negative payloads, signed zero and subnormals.
        raw = np.resize(
            np.array([0, 1, 0x7DFF, 0x7C01, 0x7F81, 0xFF81, 0x3C00, 0x8000], np.uint16), 32
        )
        inputs = {
            "raw": raw,
            "typed": raw.view(ml_dtypes.bfloat16 if dtype == "bfloat16" else dtype),
            "output": np.zeros((4, 32), np.uint16),
        }
        expected = np.broadcast_to(raw, (4, 32))
        actual = run_checked(kernel, inputs, cache_dir=tmp_path).outputs["output"]
        np.testing.assert_array_equal(actual, expected)
        gpu_inputs = dict(inputs, typed=PairedBuffer(raw, dtype)) if dtype == "bfloat16" else inputs
        gpu = gpu_runner(kernel, gpu_inputs, outputs=("output",), arch="sm_100a")["output"]
        np.testing.assert_array_equal(gpu, expected)


def test_selp_does_not_make_source_evaluation_lazy(tmp_path):
    @T.prim_func
    def kernel(source: T.Buffer((32,), "uint32"), output: T.Buffer((32,), "uint32")):
        T.device_entry()
        _warp = T.warp_id([1])
        lane = T.lane_id([32])
        T.ptx.selp.b32(output[lane], source[32], source[lane], T.bool(False))

    inputs = dict(source=np.arange(32, dtype=np.uint32), output=np.zeros(32, np.uint32))
    assert_rejected(kernel, inputs, "out-of-bounds", cache_dir=tmp_path)
