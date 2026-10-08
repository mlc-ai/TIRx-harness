"""Independent numerical oracles and CUDA lowering agree with native waits."""

import numpy as np
import pytest
import tvm
from tvm.script import tirx as T

from tests.numsim.microtests.harness import (
    NUMSIM_GPU_MARK, require_numsim_gpu, run_gpu_primfunc,
)
from tests.numsim.support.execution import run_checked
from tests.numsim.support.three_way import run_three_way_case
from tests.numsim.support.wait_until import handoff_case, initial_case


@NUMSIM_GPU_MARK
@pytest.mark.parametrize("dtype", ["int32", "uint32", "int64", "uint64"])
@pytest.mark.parametrize("kind", ["handoff", "initial"])
def test_wait_matches_gpu_and_reference(pytestconfig, tmp_path, dtype, kind):
    require_numsim_gpu(pytestconfig)
    # Select stays inside the CUDA wait macro. TVM's if_then_else currently
    # hoists a temporary before the first load, freezing a false predicate.
    case = handoff_case(dtype) if kind == "handoff" else initial_case(dtype)
    run_three_way_case(case, cache_dir=tmp_path).require_ok()


@NUMSIM_GPU_MARK
@pytest.mark.parametrize(("alias", "table_size"), [(False, 2), (True, 2), (False, 65)])
def test_indexed_predicate_matches_gpu(pytestconfig, tmp_path, alias, table_size):
    from tests.numsim.support.wait_until import indexed_predicate_case

    require_numsim_gpu(pytestconfig)
    run_three_way_case(indexed_predicate_case(alias, table_size), cache_dir=tmp_path).require_ok()


@NUMSIM_GPU_MARK
@pytest.mark.parametrize(
    "operation",
    ["bitwise_and", "bitwise_or", "bitwise_xor", "bitwise_not", "shift_left", "shift_right"],
)
def test_bitwise_wait_predicate_rechecks_loaded_value(pytestconfig, tmp_path, operation):
    require_numsim_gpu(pytestconfig)
    extra = "" if operation == "bitwise_not" else ", T.int32(3)"
    kernel = tvm.script.from_source(
        f'''
@T.prim_func
def wait(state: T.Buffer((32,), "int32"), out: T.Buffer((32,), "int32")):
    T.device_entry()
    lane = T.thread_id([32])
    seen = T.alloc_local((1,), "int32")
    seen[0] = 999
    T.cuda.wait_until(seen[0], state.ptr_to([lane]),
        lambda current: T.{operation}(current{extra}) == T.{operation}(T.int32(22){extra}))
    out[lane] = seen[0]
''',
        {"T": T},
    )
    # Every predicate is false for 999 and true after loading the state word.
    expected = np.full(32, 22, np.int32)
    inputs = {"state": expected.copy(), "out": np.zeros(32, np.int32)}
    result = run_checked(kernel, inputs, outputs=("out",), cache_dir=tmp_path)
    gpu = run_gpu_primfunc(kernel, inputs, outputs=("out",), arch="sm_100a")
    np.testing.assert_array_equal(result.outputs["out"], expected)
    np.testing.assert_array_equal(gpu["out"], expected)
