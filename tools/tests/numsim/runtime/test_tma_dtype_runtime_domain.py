from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest
import tvm

from tirx_harness import numsim
from tvm.script import tirx as T
from tvm.script.tirx import tile as Tx


@dataclass(frozen=True)
class _TmaDtypeCase:
    dtype: str
    source: np.ndarray


def _roundtrip_kernel(dtype: str, extent: int):
    name = f"typed_tma_dtype_roundtrip_{dtype.replace('x', '_x')}"
    source = "\n".join(
        (
            "@T.prim_func",
            (
                f'def {name}(source: T.Buffer(({extent},), "{dtype}"), '
                f'output: T.Buffer(({extent},), "{dtype}")):'
            ),
            "    T.device_entry()",
            "    warp = T.warp_id([2])",
            "    lane = T.lane_id([32])",
            f'    shared = T.alloc_buffer(({extent},), "{dtype}", scope="shared")',
            '    barrier = T.alloc_buffer((1,), "uint64", scope="shared")',
            "    if (warp == 0) and (lane == 0):",
            "        T.ptx.mbarrier.init.shared.b64(T.address_of(barrier[0]), 1)",
            "    T.ptx.fence.proxy.async_.shared__cta()",
            "    T.ptx.fence.mbarrier_init.release.cluster()",
            "    T.cuda.cta_sync()",
            "    if (warp == 0) and (lane == 0):",
            (
                '        Tx.copy_async(shared[:], source[:], dispatch="tma_auto", '
                "mbar=T.address_of(barrier[0]))"
            ),
            "        T.ptx.mbarrier.arrive.expect_tx.shared.b64(T.address_of(barrier[0]), 64)",
            "    if (warp == 1) and (lane == 0):",
            "        T.cuda.mbarrier_wait(T.address_of(barrier[0]), 0)",
            '        Tx.copy_async(output[:], shared[:], dispatch="tma_auto")',
            "        T.ptx.cp.async_.bulk.commit_group()",
            "        T.ptx.cp.async_.bulk.wait_group(0)",
        )
    )
    return tvm.script.from_source(source, extra_vars={"T": T, "Tx": Tx})


_SUPPORTED_CASES = (
    _TmaDtypeCase(
        "float64",
        np.array(
            [
                0.0,
                -0.0,
                1.25,
                -3.5,
                np.inf,
                -np.inf,
                np.float64(2**-100),
                np.float64(2**100),
            ],
            dtype=np.float64,
        ),
    ),
    _TmaDtypeCase("int8", np.arange(-32, 32, dtype=np.int8)),
    _TmaDtypeCase(
        "int32",
        (np.arange(16, dtype=np.int32) * np.int32(0x10203)) ^ np.int32(-0x1234567),
    ),
    _TmaDtypeCase("uint8", np.arange(64, dtype=np.uint8) ^ np.uint8(0xA5)),
    _TmaDtypeCase(
        "uint16",
        np.arange(32, dtype=np.uint16) * np.uint16(257) ^ np.uint16(0xA55A),
    ),
)

_REJECTED_REASONS = (
    (
        "bool",
        "production TensorMap encoder has no bool element ABI",
        "dtype=bool without a production TensorMap element ABI",
    ),
    (
        "float4_e2m1fn",
        "no exact CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN16B shared-layout model",
        "dtype=float4_e2m1fn without an exact 16U4_ALIGN16B shared-layout model",
    ),
    (
        "float8_e8m0fnu",
        "production TensorMap encoder has no float8_e8m0fnu element ABI",
        "dtype=float8_e8m0fnu without a production TensorMap element ABI",
    ),
    (
        "int16",
        "production TensorMap encoder has no signed int16 element ABI",
        "dtype=int16 without a signed production TensorMap element ABI",
    ),
    (
        "uint32x2",
        "production TensorMap encoder requires a scalar dtype with lanes=1",
        "dtype=uint32x2 with lanes=2 where the production encoder requires lanes=1",
    ),
)


@pytest.mark.parametrize("case", _SUPPORTED_CASES, ids=lambda case: case.dtype)
def test_typed_tma_supported_dtype_domain_preserves_values_and_bytes(tmp_path, case):
    source = case.source
    output = np.zeros_like(source)
    kernel = _roundtrip_kernel(case.dtype, source.size)
    module = numsim.transpile(kernel, cache_dir=tmp_path)
    result = numsim.Engine().run(module, {"source": source, "output": output})

    def check() -> None:
        actual = np.ascontiguousarray(result.outputs["output"])
        expected = np.ascontiguousarray(source)
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(actual.view(np.uint8), expected.view(np.uint8))

    check()


def test_typed_tma_rejects_unmodeled_production_dtypes(tmp_path):
    observed = set()
    for dtype, reason, member in _REJECTED_REASONS:
        extent = {
            "bool": 512,
            "float4_e2m1fn": 128,
            "float8_e8m0fnu": 64,
            "int16": 32,
            "uint32x2": 8,
        }[dtype]
        kernel = _roundtrip_kernel(dtype, extent)
        with pytest.raises(numsim.UnmodeledTIRxFormError, match=reason) as caught:
            numsim.transpile(kernel, cache_dir=tmp_path / dtype)
        assert caught.value.target_id == "tile:tirx.tile.copy_async"
        observed.add(member)
    assert observed == {member for _dtype, _reason, member in _REJECTED_REASONS}
