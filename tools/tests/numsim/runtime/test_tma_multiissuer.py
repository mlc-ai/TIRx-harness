"""Thread-local TMA coordinates and multicast completion targets.

Per-lane completion uses raw PTX polling: the pinned TVM cuda.mbarrier_wait
helper emits bra.uni and must not poll independently completing lane barriers.
"""

import numpy as np
import pytest
import tvm
from tvm.script import tirx as T

from tests.numsim.support.execution import run_checked
from tirx_harness import numsim, racecheck

TMA_MULTIISSUER_CASES = [
    (False, "cta"),
    (False, "cluster"),
    (False, "multicast"),
    (False, "pair"),
    (True, "cta"),
    (True, "multicast"),
    (True, "pair"),
]


def tma_multiissuer_case(gather, route, sparse=False, *, wait=True, overlap=False):
    multicast = route in {"multicast", "pair"}
    group = 2 if route == "pair" else 1
    ctas = 4 if group == 2 else 2 if multicast else 1
    selected = f"(lane % 3 == 0 or cta // {group} == lane % 2)" if multicast else "True"
    # Issuer selection must not select only the broadcast branch of the mask.
    active = "lane % 4 != 3" if sparse else "True"
    # Mix single-target and broadcast lanes; paired multicast aggregates two
    # target payloads on the even CTA's barrier without dropping either copy.
    mask = (
        f"T.Cast('uint16', T.Select(lane % 3 == 0, {(1 << ctas) - 1}, "
        f"{(1 << group) - 1} << (lane % 2 * {group})))"
    )
    instruction = f"cp.async.bulk.tensor.{2 if gather else 1}d.shared::{'cta' if route == 'cta' else 'cluster'}.global"
    instruction += ".tile::gather4" if gather else ".tile"
    instruction += ".mbarrier::complete_tx::bytes"
    if multicast:
        instruction += ".multicast::cluster"
    instruction += f".cta_group::{group}"
    coordinates = "0, lane * 4 + 2, lane * 4, lane * 4 + 3, lane * 4 + 1" if gather else "lane * 8"
    payload = 128 if gather else 32
    kernel = tvm.script.from_source(
        f"""
@T.prim_func
def kernel(tensor_map: T.TensorMap(), output: T.Buffer(({ctas}, 32, 32), "float32")):
    T.device_entry()
    _cluster = T.cluster_id([1])
    cta = T.cta_id_in_cluster([{ctas}])
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((1024,), "float32", scope="shared", align=128)
    barriers = T.alloc_buffer((32,), "uint64", scope="shared")
    ready = T.alloc_local((1,), "uint32")
    for i in T.serial(32):
        shared[lane * 32 + i] = T.float32(-7)
    if cta % {group} == 0:
        T.ptx.mbarrier.init.shared.b64(barriers.ptr_to([lane]), 1)
    T.ptx.fence.proxy.async_.shared__cta()
    T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.cluster_sync()
    if ({active}) and ({selected}) and cta % {group} == 0:
        T.ptx.mbarrier.arrive.expect_tx.shared.b64(barriers.ptr_to([lane]), {payload * group})
    T.cuda.cluster_sync()
    if cta == 0:
        T.ptx["{instruction}"](shared.ptr_to([{0 if overlap else "lane * 32"}]),
            T.address_of(tensor_map), {coordinates}, barriers.ptr_to([lane]){", " + mask if multicast else ""}, pred={active})
    if {wait} and ({active}) and ({selected}) and cta % {group} == 0:
        ready[0] = T.uint32(0)
        while ready[0] == 0:
            T.ptx.mbarrier.try_wait.parity.shared.b64(
                ready[0], barriers.ptr_to([lane]), T.uint32(0), T.uint32(1))
    T.cuda.cluster_sync()
    for i in T.serial(32):
        output[cta, lane, i] = shared[lane * 32 + i]
""",
        {"T": T},
    )
    source = np.arange(1024 if gather else 256, dtype=np.float32) + np.float32(0.25)
    if gather:
        source = source.reshape(128, 8)
        metadata = dict(
            global_shape=(8, 128), global_strides=(32,), box_shape=(8, 1), element_strides=(1, 1)
        )
    else:
        metadata = dict(
            global_shape=(256,), global_strides=(), box_shape=(8,), element_strides=(1,)
        )
    expected = np.full((ctas, 32, 32), -7, np.float32)
    for lane in range(32):
        if sparse and lane % 4 == 3:
            continue
        values = (
            source[lane * 4 + np.array([2, 0, 3, 1])].reshape(-1)
            if gather
            else source[lane * 8 : lane * 8 + 8]
        )
        for cta in range(ctas):
            if multicast and lane % 3 != 0 and cta // group != lane % 2:
                continue
            expected[cta, 0 if overlap else lane, : values.size] = values
    return kernel, source, metadata, expected


@pytest.mark.parametrize("gather,route", TMA_MULTIISSUER_CASES)
def test_tma_multiissuer_coordinates_and_target_completion(gather, route, tmp_path):
    # The sparse issuer mask covers inactive lanes on every route. Keep one
    # all-active control instead of rerunning every layout without that mask.
    for sparse in (False, True) if (gather, route) == (False, "cta") else (True,):
        kernel, source, metadata, expected = tma_multiissuer_case(gather, route, sparse)
        args = {
            "tensor_map": numsim.TensorMap(base=source, **metadata).numpy(),
            "output": np.zeros_like(expected),
        }
        result = run_checked(kernel, args, cache_dir=tmp_path)
        np.testing.assert_array_equal(result.outputs["output"], expected)


def test_tma_multiissuer_requires_wait_and_disjoint_destinations():
    for wait, overlap in [(False, False), (True, True)]:
        kernel, source, metadata, expected = tma_multiissuer_case(
            False, "cta", wait=wait, overlap=overlap
        )
        args = {
            "tensor_map": numsim.TensorMap(base=source, **metadata).numpy(),
            "output": np.zeros_like(expected),
        }
        report = racecheck(kernel, args)
        assert report.verdict == "error", report.format()
        kinds = {"write_write"} if overlap else {"read_write", "write_read"}
        assert any(f.kind == "data_race" and f.details["access_pair"] in kinds for f in report.findings), (
            report.format()
        )
