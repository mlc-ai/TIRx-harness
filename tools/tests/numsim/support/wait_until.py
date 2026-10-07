"""Small wait kernels shared by native and live-GPU semantic tests."""

import numpy as np
import tvm
from tvm.script import tirx as T

from tirx_harness.numsim.cases import ComparisonSpec, NumSimCase


def handoff_case(dtype="int32", scope="gpu", release=True, backoff=0):
    suffix = dtype.replace("uint", "u").replace("int", "s")
    order = "release" if release else "relaxed"
    kernel = tvm.script.from_source(
        f'''
@T.prim_func
def handoff(state: T.Buffer((1,), "{dtype}"), payload: T.Buffer((1,), "int32"),
            out: T.Buffer((2,), "int32")):
    T.device_entry()
    cta = T.cta_id([2])
    lane = T.thread_id([32])
    seen = T.alloc_local((1,), "{dtype}")
    published = T.alloc_local((1,), "{dtype}")
    if lane == 0:
        if cta == 0:
            payload[0] = 73
            published[0] = T.{dtype}(7)
            T.ptx["st.{order}.{scope}.global.{suffix}"](state.ptr_to([0]), published[0])
        else:
            seen[0] = T.{dtype}(7)
            T.cuda.wait_until(seen[0], state.ptr_to([0]),
                              lambda current: current == T.{dtype}(7),
                              scope="gpu", backoff_ns={backoff})
            out[0] = payload[0]
            out[1] = T.Cast("int32", seen[0])
''',
        {"T": T},
    )
    return NumSimCase(
        kernel=kernel,
        args={
            "state": np.zeros(1, dtype),
            "payload": np.zeros(1, np.int32),
            "out": np.zeros(2, np.int32),
        },
        outputs=("out",),
        reference=lambda: {"out": np.array([73, 7], np.int32)},
        comparisons={"out": ComparisonSpec(atol=0, rtol=0)},
    )


def initial_case(dtype="int32", ptx_type=None, predicate_op="Select"):
    extra = "" if ptx_type is None else f", ptx_type={ptx_type!r}"
    kernel = tvm.script.from_source(
        f'''
@T.prim_func
def initial(state: T.Buffer((32,), "{dtype}"), out: T.Buffer((32,), "{dtype}")):
    T.device_entry()
    T.cta_id([1])
    lane = T.thread_id([32])
    seen = T.alloc_local((1,), "{dtype}")
    target = T.alloc_local((1,), "{dtype}")
    seen[0] = T.{dtype}(999)
    target[0] = T.Cast("{dtype}", lane + 17)
    if lane < 17:
        T.cuda.wait_until(seen[0], state.ptr_to([lane]),
                          lambda current: T.{predicate_op}(lane % 2 == 0,
                              current // T.{dtype}(2) == target[0] // T.{dtype}(2),
                              (current & T.{dtype}(255)) == target[0]){extra})
        out[lane] = seen[0]
''',
        {"T": T},
    )
    expected = np.zeros(32, dtype)
    expected[:17] = np.arange(17, 34, dtype=dtype)
    return NumSimCase(
        kernel=kernel,
        args={"state": np.arange(17, 49, dtype=dtype), "out": np.zeros(32, dtype)},
        outputs=("out",),
        reference=lambda: {"out": expected},
        comparisons={"out": ComparisonSpec(atol=0, rtol=0)},
    )


def indexed_predicate_case(alias=False, table_size=2):
    if table_size < 2:
        raise ValueError("table_size must leave room for distinct false and true entries")
    candidate = 1 if alias else table_size - 1
    predicate = "seen[lane % 2] == 1" if alias else "table[current] == 1"
    table_initialization = "\n".join(
        f"    table[{index}] = {int(index == candidate)}" for index in range(table_size)
    )
    kernel = tvm.script.from_source(
        f"""
@T.prim_func
def indexed(state: T.Buffer((32,), "int32"), out: T.Buffer((32,), "int32")):
    T.device_entry()
    T.cta_id([1])
    lane = T.thread_id([32])
    seen = T.alloc_local((2,), "int32")
    table = T.alloc_local(({table_size},), "int32")
    seen[1] = 1
{table_initialization}
    T.cuda.wait_until(seen[0], state.ptr_to([lane]), lambda current: {predicate})
    out[lane] = seen[0]
""",
        {"T": T},
    )
    return NumSimCase(
        kernel=kernel,
        args={
            "state": np.full(32, candidate, np.int32),
            "out": np.zeros(32, np.int32),
        },
        outputs=("out",),
        reference=lambda: {"out": np.full(32, candidate, np.int32)},
        comparisons={"out": ComparisonSpec(atol=0, rtol=0)},
    )
