from __future__ import annotations

import numpy as np
import tvm
from ml_dtypes import bfloat16
from tvm.script import tirx as T

from tirx_harness import numsim
from tests.numsim.support.runtime_domains import (
    CUDA_ATOMIC_ADD_PACKED_GLOBAL_DTYPES,
    CUDA_ATOMIC_ADD_PACKED_SHARED_DTYPES,
    CUDA_ATOMIC_ADD_SCALAR_DTYPES,
    CUDA_ATOMIC_CAS_DTYPES,
)


_ADD_DTYPES = (
    *CUDA_ATOMIC_ADD_SCALAR_DTYPES,
    *CUDA_ATOMIC_ADD_PACKED_GLOBAL_DTYPES,
)
_SCALAR_NUMPY_DTYPES = {
    "int32": np.int32,
    "uint16": np.uint16,
    "uint32": np.uint32,
    "uint64": np.uint64,
    "float16": np.float16,
    "bfloat16": bfloat16,
    "float32": np.float32,
    "float64": np.float64,
}
_PACKED_FLOAT_BASE = {
    "float16x2": np.float16,
    "bfloat16x2": bfloat16,
    "float32x2": np.float32,
    "float32x4": np.float32,
}


def _make_atomic_full_domain():
    parameters = []
    declarations = []
    initialization = []
    statements = []
    for dtype in _ADD_DTYPES:
        name = dtype.replace("_", "")
        parameters.extend(
            (
                f'    add_global_{name}: T.Buffer((1,), "{dtype}"),',
                f'    add_value_{name}: T.Buffer((1,), "{dtype}"),',
            )
        )
        statements.append(
            f"    T.evaluate(T.cuda.atomic_add(add_global_{name}.ptr_to([0]), add_value_{name}[0]))"
        )
        if dtype in (
            *CUDA_ATOMIC_ADD_SCALAR_DTYPES,
            *CUDA_ATOMIC_ADD_PACKED_SHARED_DTYPES,
        ):
            declarations.append(
                f'    add_shared_{name} = T.alloc_buffer((1,), "{dtype}", scope="shared")'
            )
            initialization.extend(
                (
                    "    if lane == 0:",
                    f"        add_shared_{name}[0] = add_value_{name}[0]",
                )
            )
            statements.append(
                f"    T.evaluate(T.cuda.atomic_add(add_shared_{name}.ptr_to([0]), "
                f"add_value_{name}[0]))"
            )
    for dtype in CUDA_ATOMIC_CAS_DTYPES:
        name = dtype.replace("_", "")
        parameters.extend(
            (
                f'    cas_global_{name}: T.Buffer((1,), "{dtype}"),',
                f'    cas_compare_{name}: T.Buffer((1,), "{dtype}"),',
                f'    cas_value_{name}: T.Buffer((1,), "{dtype}"),',
            )
        )
        statements.append(
            "    T.evaluate(T.cuda.atomic_cas("
            f"cas_global_{name}.ptr_to([0]), cas_compare_{name}[0], cas_value_{name}[0]))"
        )
    source = "\n".join(
        (
            "@T.prim_func",
            "def atomic_full_domain(",
            *parameters,
            "):",
            "    T.device_entry()",
            "    _warp = T.warp_id([1])",
            "    lane = T.lane_id([32])",
            *declarations,
            *initialization,
            "    T.cuda.warp_sync()",
            *statements,
        )
    )
    return tvm.script.from_source(source, extra_vars={"T": T})


atomic_full_domain = _make_atomic_full_domain()


def _packed_float_array(dtype: str, value: int) -> np.ndarray:
    data_type = tvm.DataType(dtype)
    logical = np.full(
        (1, data_type.lanes),
        value,
        dtype=_PACKED_FLOAT_BASE[dtype],
    )
    byte_width = data_type.bits * data_type.lanes // 8
    carrier = {4: np.uint32, 8: np.uint64, 16: np.dtype("V16")}[byte_width]
    return logical.copy().view(carrier).reshape(-1)


def _physical_array(dtype: str, value: int) -> np.ndarray:
    data_type = tvm.DataType(dtype)
    if data_type.lanes == 1:
        return np.full(1, value, dtype=_SCALAR_NUMPY_DTYPES[dtype])
    if dtype in _PACKED_FLOAT_BASE:
        return _packed_float_array(dtype, value)
    assert data_type.bits * data_type.lanes == 128
    raw = np.zeros(16, dtype=np.uint8) if value == 0 else np.arange(1, 17, dtype=np.uint8)
    return raw.view(np.dtype("V16"))


def _binding(
    dtype: str,
    value: int,
) -> np.ndarray:
    return _physical_array(dtype, value)


def _raw_bytes(value: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(value).view(np.uint8).reshape(-1)


def test_every_cuda_atomic_add_and_cas_form_executes_with_exact_bits(tmp_path):
    module = numsim.transpile(atomic_full_domain, cache_dir=tmp_path)
    args: dict[str, np.ndarray] = {}
    expected: dict[str, np.ndarray] = {}
    for dtype in _ADD_DTYPES:
        name = dtype.replace("_", "")
        output_name = f"add_global_{name}"
        args[output_name] = _binding(
            dtype,
            0,
        )
        args[f"add_value_{name}"] = _binding(
            dtype,
            1,
        )
        expected[output_name] = _physical_array(dtype, 32)
    for dtype in CUDA_ATOMIC_CAS_DTYPES:
        name = dtype.replace("_", "")
        output_name = f"cas_global_{name}"
        args[output_name] = _binding(
            dtype,
            0,
        )
        args[f"cas_compare_{name}"] = _binding(
            dtype,
            0,
        )
        args[f"cas_value_{name}"] = _binding(
            dtype,
            1,
        )
        expected[output_name] = _physical_array(dtype, 1)

    result = numsim.Engine().run(module, args)

    def check():
        assert set(result.outputs) == set(args)
        for name, value in expected.items():
            np.testing.assert_array_equal(
                _raw_bytes(result.outputs[name]),
                _raw_bytes(value),
                err_msg=name,
            )
        assert result.stats["task_count"] == 1
        assert result.stats["completed_task_count"] == 1

    check()
