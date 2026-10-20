"""Memory-ordering checks for the benchmarked full-scale GEMM/all-reduce port."""

from __future__ import annotations

import pytest
import tvm
import tvm_ffi

from tests.numsim.support import multimem_gemm_all_reduce as full
from tirx_harness.numsim.checkers import _run_racecheck as racecheck


@pytest.fixture(scope="module")
def kernel_cache(tmp_path_factory):
    return tmp_path_factory.mktemp("racecheck-full-multimem-gemm")


@pytest.mark.parametrize("world", (2, 4))
@pytest.mark.parametrize("protocol", ("cutlass", "flashinfer"))
@pytest.mark.parametrize("use_tma_store", (True, False))
def test_full_gemm_system_publication_is_clean(world, protocol, use_tma_store, kernel_cache):
    shape = full.Shape(
        world=world,
        protocol=protocol,
        config={**full.Shape().config, "use_tma_store": use_tma_store},
    )
    bindings, _ = full.rank_inputs(shape)

    report = racecheck(full.kernel(shape), inputs=bindings, cache_dir=kernel_cache)

    assert report.native_payload["incomplete"] == []
    assert report.native_payload["findings"] == []
    assert report.verdict == "clean", report.format()


@pytest.mark.parametrize("protocol", ("cutlass", "flashinfer"))
def test_full_gemm_origin_signaling_leaves_cross_rank_conflicts(protocol, kernel_cache):
    shape = full.Shape(protocol=protocol)
    bindings, _ = full.rank_inputs(shape)

    report = racecheck(
        full.kernel(shape, ordering="origin"), inputs=bindings, cache_dir=kernel_cache
    )

    assert report.native_payload["incomplete"] == []
    assert report.verdict == "error", report.format()
    plan = shape.plan()
    warps_per_rank = plan.clusters * plan.cluster_size * (6 + plan.comm_warps)
    races = [
        finding for finding in report.native_payload["findings"] if finding["kind"] == "data_race"
    ]
    assert any(
        finding["prior"]["operation"]["global_warp_id"] // warps_per_rank
        != finding["current"]["operation"]["global_warp_id"] // warps_per_rank
        and finding["overlap"]["byte_len"] >= 16
        for finding in races
    ), report.format()


@pytest.mark.parametrize("protocol", ("cutlass", "flashinfer"))
@pytest.mark.parametrize("use_tma_store", (True, False))
@pytest.mark.parametrize("use_2cta", (True, False))
def test_full_gemm_multiple_ctas_publish_every_writer(
    protocol, use_tma_store, use_2cta, kernel_cache,
):
    shape = full.multi_cta_shape(protocol, use_tma_store, use_2cta)
    bindings, _ = full.rank_inputs(shape)

    report = racecheck(full.kernel(shape), inputs=bindings, cache_dir=kernel_cache)

    assert report.native_payload["incomplete"] == []
    assert report.native_payload["findings"] == []
    assert report.verdict == "clean", report.format()


@pytest.mark.parametrize("protocol", ("cutlass", "flashinfer"))
@pytest.mark.parametrize("use_tma_store", (True, False))
def test_full_gemm_multiple_ctas_require_the_alias_bridge(protocol, use_tma_store, kernel_cache):
    shape = full.Shape(
        protocol=protocol,
        config={**full.Shape().config, "max_active_clusters": 2, "use_tma_store": use_tma_store},
    )
    function = full.kernel(shape)
    wanted = ("proxy", "alias")

    removed = 0

    def drop_fence(statement):
        nonlocal removed
        call = statement.value
        if not isinstance(call, tvm.ir.Call) or getattr(call.op, "name", "") not in (
            "tirx.ptx.fence", "tirx.ptx.fence_proxy",
        ):
            return statement
        modifiers = tuple(
            argument.value for argument in call.args if isinstance(argument, tvm.ir.StringImm)
        )
        if modifiers[:2] != wanted:
            return statement
        removed += 1
        return tvm.tirx.Evaluate(0, span=statement.span)

    broken = tvm_ffi.structural_map(function, (tvm.tirx.Evaluate, drop_fence))
    assert removed > 0
    bindings, _ = full.rank_inputs(shape)

    report = racecheck(broken, inputs=bindings, cache_dir=kernel_cache)

    assert report.native_payload["incomplete"] == []
    assert report.verdict == "error", report.format()
    assert any(finding["kind"] == "data_race" for finding in report.native_payload["findings"])
