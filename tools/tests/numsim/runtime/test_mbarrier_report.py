"""Copy validity reports share the barrier's phase and completion contract."""

import numpy as np
import pytest
import tvm
from tvm.script import tirx as T

from tests.numsim.runtime.test_raw_tma_codegen import _tensor_map
from tests.numsim.support.execution import run_checked


def report_kernel(pattern, *, cluster=False, layout=1, tensor=False):
    family = ".tensor.1d" if tensor else ""
    queries = []
    for parity in (False, True):
        for action, hint in (("test_wait", False), ("try_wait", False), ("try_wait", True)):
            for value in (False, True):
                slot = len(queries)
                operands = "ready[0], report[0], " + ("value[0], " if value else "")
                operands += "barrier.ptr_to([0]), " + ("T.uint32(phase)" if parity else "state[0]")
                operands += ", T.uint32(1)" if hint else ""
                suffix = ".parity" if parity else ""
                queries.append(
                    f'T.ptx["mbarrier.{action}{suffix}.phase_type::primary.shared.b64"]({operands})\n'
                    f"out[phase, {slot}] = ready[0] + 2 * report[0]"
                    + (" + T.Cast('uint32', value[0]) * 4" if value else "")
                )
    query_source = "\n            ".join(q.replace("\n", "\n            ") for q in queries)
    report = "disabled" if pattern == "disabled" else f"validity::{pattern}"
    parameter = "input_map: T.TensorMap()" if tensor else 'source: T.Buffer((32,), "uint8")'
    source_operands = (
        "T.address_of(input_map), phase * 16"
        if tensor
        else "source.ptr_to([phase * 16]), T.uint32(16)"
    )
    return tvm.script.from_source(
        f"""
@T.prim_func
def kernel({parameter}, out: T.Buffer((2, 12), "uint32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    barrier = T.alloc_buffer((1,), "uint64", scope="shared")
    shared = T.alloc_buffer((16,), "uint8", scope="shared", align=128)
    state = T.alloc_local((1,), "uint64")
    ready = T.alloc_local((1,), "uint32")
    report = T.alloc_local((1,), "uint32")
    value = T.alloc_local((1,), "uint8")
    if lane == 0:
        T.ptx["mbarrier.init.layout::v{layout}.shared.b64"](barrier.ptr_to([0]), 1)
    T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.cta_sync()
    for phase in T.serial(2):
        if lane == 0:
            T.ptx.mbarrier.arrive.expect_tx.shared.b64(state[0], barrier.ptr_to([0]), 16)
            T.ptx["cp.async.bulk{family}.shared::{"cluster" if cluster else "cta"}.global.mbarrier::complete_tx::bytes.mbarrier::report::{report}"](
                shared.ptr_to([0]), {source_operands}, barrier.ptr_to([0]))
            T.cuda.mbarrier_wait(barrier.ptr_to([0]), phase)
            {query_source}
""",
        {"T": T},
    )


@pytest.mark.parametrize(
    "pattern",
    [
        "disabled",
        "per_element::ff",
        "per_16bytes::80000000",
        "per_16bytes::8000",
        "per_16bytes::80",
        "per_16bytes::8",
    ],
)
def test_report_queries_and_phase_reset(pattern, tmp_path):
    kernel = report_kernel(
        pattern, cluster=pattern.endswith("80"), layout=0 if pattern == "disabled" else 1
    )
    source = np.zeros(32, np.uint8)
    if pattern != "disabled":
        bits = int(pattern.rsplit("::", 1)[-1], 16)
        source[:4] = np.frombuffer(bits.to_bytes(4, "little"), np.uint8)
    args = {"source": source, "out": np.zeros((2, 12), np.uint32)}
    result = run_checked(kernel, args, cache_dir=tmp_path)
    expected = np.ones((2, 12), np.uint32)
    if pattern != "disabled":
        expected[0, :] = 3
    np.testing.assert_array_equal(result.outputs["out"], expected)


@pytest.mark.parametrize("cluster", [False, True])
def test_tensor_copy_report(cluster, tmp_path):
    source = np.zeros(32, np.uint8)
    source[7] = 255
    descriptor, _ = _tensor_map(source, global_shape=(32,), global_strides=(), box_shape=(16,))
    args = {"input_map": descriptor, "out": np.zeros((2, 12), np.uint32)}
    kernel = report_kernel("per_element::ff", tensor=True, cluster=cluster)
    result = run_checked(kernel, args, cache_dir=tmp_path)
    expected = np.ones((2, 12), np.uint32)
    expected[0, :] = 3
    np.testing.assert_array_equal(result.outputs["out"], expected)


def gather4_report_case(pattern, matched, override=False):
    kernel = tvm.script.from_source(
        f"""
@T.prim_func
def kernel(input_map: T.TensorMap(), out: T.Buffer((17,), "uint32"){', replacement: T.Buffer((32768,), "uint32")' if override else ""}):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((16,), "uint32", scope="shared", align=128)
    barrier = T.alloc_buffer((1,), "uint64", scope="shared")
    ready = T.alloc_local((1,), "uint32")
    report = T.alloc_local((1,), "uint32")
    if lane == 0:
        T.ptx["mbarrier.init.layout::v1.shared.b64"](barrier.ptr_to([0]), 1)
    T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.cta_sync()
    if lane == 0:
        T.ptx.mbarrier.arrive.expect_tx.shared.b64(barrier.ptr_to([0]), 64)
        T.ptx["cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4.mbarrier::complete_tx::bytes.mbarrier::report::validity::{pattern}{".override::global_address" if override else ""}"](
            shared.ptr_to([0]), T.address_of(input_map){', T.reinterpret("uint64", replacement.ptr_to([0]))' if override else ""}, 0, 3, 1, 3, 9, barrier.ptr_to([0]))
        T.cuda.mbarrier_wait(barrier.ptr_to([0]), 0)
        T.ptx["mbarrier.test_wait.parity.phase_type::primary.shared.b64"](
            ready[0], report[0], barrier.ptr_to([0]), 0)
        out[16] = ready[0] + 2 * report[0]
    T.cuda.warp_sync()
    if lane < 16:
        out[lane] = shared[lane]
""",
        {"T": T},
    )
    source = np.arange(16, dtype=np.uint32).reshape(4, 4)
    source[0, :] = 0xFFFFFFFF  # This row is not copied.
    if pattern.endswith("80000000"):
        source[1, 1] = 0x80000000  # Not the sampled element in this 16B chunk.
    if matched:
        source[3, 0] = 0x80000000 if pattern.endswith("80000000") else 0xFFFFFFFF
    original = (
        np.full_like(source, 0x80000000 if pattern.endswith("80000000") else 0xFFFFFFFF)
        if override
        else source
    )
    descriptor, _ = _tensor_map(
        original, global_shape=(4, 4), global_strides=(16,), box_shape=(4, 1)
    )
    args = {"input_map": descriptor, "out": np.zeros(17, np.uint32)}
    if override:
        args["replacement"] = np.zeros(32768, np.uint32)
        args["replacement"][:16] = source.ravel()
    expected = np.concatenate((source[[3, 1, 3]].ravel(), np.zeros(4, np.uint32)))
    return kernel, args, expected


@pytest.mark.parametrize("pattern", ["per_element::ff", "per_16bytes::80000000"])
@pytest.mark.parametrize("matched", [False, True])
def test_gather4_report_samples_only_selected_source_rows(pattern, matched, tmp_path):
    for override in (False, True):
        kernel, args, expected = gather4_report_case(pattern, matched, override)
        result = run_checked(kernel, args, cache_dir=tmp_path)
        np.testing.assert_array_equal(result.outputs["out"][:16], expected)
        assert result.outputs["out"][16] == 1 + 2 * matched


@pytest.mark.parametrize("tensor,scope", [(False, None), (False, "cluster"), (True, None)])
def test_multicast_report_targets(tensor, scope, tmp_path):
    parameter = "input_map: T.TensorMap()" if tensor else 'source: T.Buffer((16,), "uint8")'
    operands = "T.address_of(input_map), 0" if tensor else "source.ptr_to([0]), T.uint32(16)"
    family = ".tensor.1d" if tensor else f".relaxed.{scope}" if scope else ""
    element = ".b128" if scope else ""
    kernel = tvm.script.from_source(
        f"""
@T.prim_func
def kernel({parameter}, out: T.Buffer((3,), "uint32")):
    T.device_entry()
    _cluster = T.cluster_id([1])
    cta = T.cta_id_in_cluster([3])
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    barrier = T.alloc_buffer((1,), "uint64", scope="shared")
    shared = T.alloc_buffer((16,), "uint8", scope="shared", align=128)
    ready = T.alloc_local((1,), "uint32")
    report = T.alloc_local((1,), "uint32")
    if lane == 0:
        T.ptx["mbarrier.init.layout::v1.shared.b64"](barrier.ptr_to([0]), 1)
    T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.cluster_sync()
    if lane == 0:
        if cta != 1:
            T.ptx.mbarrier.arrive.expect_tx.shared.b64(barrier.ptr_to([0]), 16)
        else:
            T.ptx.mbarrier.arrive.shared.b64(barrier.ptr_to([0]))
    if cta == 0 and lane == 0:
        T.ptx["cp.async.bulk{family}.shared::cluster.global.mbarrier::complete_tx::bytes.mbarrier::report::validity::per_element::ff.multicast::cluster::32b{element}"](
            shared.ptr_to([0]), {operands}, barrier.ptr_to([0]), T.uint32(5))
    if lane == 0:
        T.cuda.mbarrier_wait(barrier.ptr_to([0]), 0)
        T.ptx["mbarrier.test_wait.parity.phase_type::primary.shared.b64"](ready[0], report[0], barrier.ptr_to([0]), 0)
        out[cta] = ready[0] + 2 * report[0]
    T.cuda.cluster_sync()
""",
        {"T": T},
    )
    source = np.full(16, 255, np.uint8)
    args = {"out": np.zeros(3, np.uint32)}
    if tensor:
        args["input_map"], _ = _tensor_map(
            source, global_shape=(16,), global_strides=(), box_shape=(16,)
        )
    else:
        args["source"] = source
    result = run_checked(kernel, args, cache_dir=tmp_path)
    np.testing.assert_array_equal(result.outputs["out"], [3, 1, 3])
