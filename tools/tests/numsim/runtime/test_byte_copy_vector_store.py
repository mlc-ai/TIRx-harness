from __future__ import annotations

import numpy as np
import pytest

from tirx_harness import numsim
from tirx_harness.numsim.errors import NumSimExecutionError
from tirx_harness.numsim.transpiler.frontend import analyze
from tvm_tirx.script import tirx as T


@T.function
def generic_vector_store_forms(
    source_u32: T.Tensor((256,), "uint32"),
    destination_u32: T.Tensor((256,), "uint32"),
    source_f64: T.Tensor((64,), "float64"),
    destination_f64: T.Tensor((64,), "float64"),
):
    T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=32))
    lane = T.cuda.lane_id()
    base8 = lane * 8
    T.ptx.st.global_.v8.u32(
        T.address_of(destination_u32[base8]),
        source_u32[base8],
        source_u32[base8 + 1],
        source_u32[base8 + 2],
        source_u32[base8 + 3],
        source_u32[base8 + 4],
        source_u32[base8 + 5],
        source_u32[base8 + 6],
        source_u32[base8 + 7],
    )
    base2 = lane * 2
    T.ptx.st.global_.v2.f64(
        T.address_of(destination_f64[base2]),
        source_f64[base2],
        source_f64[base2 + 1],
    )


@T.function
def misaligned_vector_store(destination: T.Tensor((64,), "uint8")):
    T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=32))
    lane = T.cuda.lane_id()
    if lane == 0:
        T.ptx.st.global_.v4.b32(
            T.address_of(destination[1]),
            T.uint32(1),
            T.uint32(2),
            T.uint32(3),
            T.uint32(4),
        )


def test_vector_store_registry_is_exact():
    assert analyze(generic_vector_store_forms).unsupported == ()


def test_generic_ptx_vector_store_widths_and_types(tmp_path):
    source_u32 = np.arange(256, dtype=np.uint32) ^ np.uint32(0xA5A55A5A)
    source_f64 = np.linspace(-3.0, 5.0, 64, dtype=np.float64)
    module = numsim.transpile(generic_vector_store_forms, cache_dir=tmp_path)
    result = numsim.Engine().run(
        module,
        {
            "source_u32": source_u32,
            "destination_u32": np.zeros_like(source_u32),
            "source_f64": source_f64,
            "destination_f64": np.zeros_like(source_f64),
        },
    )
    np.testing.assert_array_equal(result.outputs["destination_u32"], source_u32)
    np.testing.assert_array_equal(result.outputs["destination_f64"], source_f64)


def test_vector_store_checks_total_access_width_alignment(tmp_path):
    module = numsim.transpile(misaligned_vector_store, cache_dir=tmp_path)
    with pytest.raises(NumSimExecutionError, match="16-byte alignment"):
        numsim.Engine().run(module, {"destination": np.zeros(64, dtype=np.uint8)})
