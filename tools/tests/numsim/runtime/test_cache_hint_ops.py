from __future__ import annotations

import numpy as np
import pytest

from tirx_harness import numsim
from tirx_harness.numsim.checkers import _run_racecheck, _run_synccheck
from tests.numsim.support.manifest import call_op_names
from tvm.script import tirx as T


CACHE_HINT_OPS = frozenset(
    (
        "tirx.ptx.applypriority",
        "tirx.ptx.applypriority_async_bulk",
        "tirx.ptx.applypriority_async_bulk_tensor",
        "tirx.ptx.cp_async_bulk_prefetch_evict_last",
        "tirx.ptx.cp_async_bulk_tensor_prefetch_evict_last",
        "tirx.ptx.prefetch_valid_addr",
        "tirx.ptx.prefetchu",
    )
)


@T.prim_func
def raw_cache_hint_family(
    source: T.Buffer((256,), "uint8"),
    input_map: T.TensorMap(),
    output: T.Buffer((32,), "uint8"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    issue = lane < 16

    T.ptx["prefetch.L1::32B.valid_addr"](source.ptr_to([0]), pred=issue)
    T.ptx["prefetchu.L1"](source.ptr_to([0]), pred=issue)
    T.ptx["applypriority.L2::evict_normal"](source.ptr_to([0]), pred=issue)
    T.ptx["cp.async.bulk.prefetch.L2.global.L2::evict_last"](
        source.ptr_to([16]), T.uint32(32), pred=issue
    )
    T.ptx["applypriority.async.bulk.bulk_group.L2::evict_normal"](
        source.ptr_to([0]), T.uint32(32), pred=issue
    )
    T.ptx["cp.async.bulk.prefetch.tensor.2d.L2.global.L2::evict_last"](
        T.address_of(input_map), T.int32(0), T.int32(0), pred=issue
    )
    T.ptx["applypriority.async.bulk.tensor.2d.bulk_group.L2::evict_normal"](
        T.address_of(input_map), T.int32(0), T.int32(0), pred=issue
    )
    T.ptx.cp.async_.bulk.commit_group()
    T.ptx.cp.async_.bulk.wait_group.read(0)
    output[lane] = source[lane + 32]


@T.prim_func
def predicated_valid_address(source: T.Buffer((256,), "uint8"), enabled: T.int32):
    T.device_entry()
    _warp = T.warp_id([1])
    T.ptx["prefetch.L1::32B.valid_addr"](source.ptr_to([256]), pred=enabled)


@T.prim_func
def invalid_scalar_cache_hint_contracts(
    source: T.Buffer((256,), "uint8"),
    apply_offset: T.int32,
    prefetch_offset: T.int32,
    prefetch_size: T.uint32,
    bulk_apply_offset: T.int32,
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    issue = lane == 0
    T.ptx["applypriority.L2::evict_normal"](source.ptr_to([apply_offset]), pred=issue)
    T.ptx["cp.async.bulk.prefetch.L2.global.L2::evict_last"](
        source.ptr_to([prefetch_offset]), prefetch_size, pred=issue
    )
    T.ptx["applypriority.async.bulk.bulk_group.L2::evict_normal"](
        source.ptr_to([bulk_apply_offset]), T.uint32(16), pred=issue
    )


@T.prim_func
def raw_gather4_cache_hint(descriptor: T.Buffer((128,), "uint8")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    T.ptx["applypriority.async.bulk.tensor.2d.global.bulk_group.tile::gather4.L2::evict_normal"](
        descriptor.ptr_to([0]),
        T.int32(0),
        T.int32(0),
        T.int32(0),
        T.int32(0),
        T.int32(0),
        pred=lane == 0,
    )
    T.ptx["cp.async.bulk.prefetch.tensor.2d.L2.global.tile::gather4.L2::evict_last"](
        descriptor.ptr_to([0]),
        T.int32(0),
        T.int32(0),
        T.int32(0),
        T.int32(0),
        T.int32(0),
        pred=lane == 0,
    )


@T.prim_func
def varying_typed_tensor_map_hint(
    first_map: T.TensorMap(), second_map: T.TensorMap(), single_lane: T.int32
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    issue = T.Select(single_lane != 0, lane == 0, T.bool(True))
    T.ptx["cp.async.bulk.prefetch.tensor.2d.L2.global.L2::evict_last"](
        T.Select(lane == 0, T.address_of(first_map), T.address_of(second_map)),
        T.int32(0),
        T.int32(0),
        pred=issue,
    )


def _tensor_map(array: np.ndarray) -> np.ndarray:
    return numsim.TensorMap(
        base=array,
        global_shape=(4, 4),
        global_strides=(16,),
        box_shape=(4, 1),
        element_strides=(1, 1),
    ).numpy()


def _cache_hint_inputs() -> dict[str, np.ndarray]:
    return {
        "source": np.arange(256, dtype=np.uint8) ^ np.uint8(0xA5),
        "input_map": _tensor_map(np.arange(16, dtype=np.float32).reshape(4, 4)),
        "output": np.zeros(32, dtype=np.uint8),
    }


def test_cache_hint_family_validates_operands_without_changing_memory(tmp_path):
    inputs = _cache_hint_inputs()
    source_before = inputs["source"].copy()

    module = numsim.transpile(raw_cache_hint_family, cache_dir=tmp_path)
    result = numsim.Engine().run(module, inputs)

    np.testing.assert_array_equal(result.outputs["output"], inputs["source"][32:64])
    np.testing.assert_array_equal(inputs["source"], source_before)
    assert result.verdict == "clean"
    assert result.diagnostics == []
    assert CACHE_HINT_OPS <= call_op_names(module.spec.kernels[0])
    assert "async_copy::prefetch_valid_address" in module.rust_source
    assert "variant::ApplyPriority" in module.rust_source
    assert "async_copy::cp_async_bulk_prefetch" in module.rust_source
    assert "variant::BulkApplyPriority" in module.rust_source
    assert "variant::TensorPrefetchEvictLast<2>" in module.rust_source
    assert "variant::TensorApplyPriority<2>" in module.rust_source


@pytest.mark.parametrize(
    "checker", [_run_synccheck, _run_racecheck], ids=["synccheck", "racecheck"]
)
def test_async_applypriority_bulk_groups_are_visible_to_checkers(tmp_path, checker):
    report = checker(
        raw_cache_hint_family,
        _cache_hint_inputs(),
        cache_dir=tmp_path,
        max_workers=1,
    )

    report.require_clean()
    # 16 issuing lanes x two applypriority instructions x two Bulk milestones.
    assert report.native_payload["stats"]["completion_operation_count"] == 64


def test_valid_address_prefetch_checks_only_predicate_selected_lanes(tmp_path):
    module = numsim.transpile(predicated_valid_address, cache_dir=tmp_path)
    inputs = {"source": np.zeros(256, dtype=np.uint8), "enabled": 0}

    result = numsim.Engine().run(module, inputs)
    assert result.verdict == "clean"

    inputs["enabled"] = 1
    with pytest.raises(numsim.NumSimExecutionError, match="out-of-bounds"):
        numsim.Engine().run(module, inputs)


def test_scalar_cache_hint_runtime_contracts_reach_the_engine(tmp_path):
    module = numsim.transpile(invalid_scalar_cache_hint_contracts, cache_dir=tmp_path)
    cases = (
        (
            {
                "apply_offset": 16,
                "prefetch_offset": 0,
                "prefetch_size": 16,
                "bulk_apply_offset": 0,
            },
            "128-byte aligned",
        ),
        (
            {
                "apply_offset": 0,
                "prefetch_offset": 0,
                "prefetch_size": 12,
                "bulk_apply_offset": 0,
            },
            "multiple of 16",
        ),
        (
            {
                "apply_offset": 0,
                "prefetch_offset": 0,
                "prefetch_size": 16,
                "bulk_apply_offset": 16,
            },
            "128-byte aligned",
        ),
        (
            {
                "apply_offset": 0,
                "prefetch_offset": 1,
                "prefetch_size": 16,
                "bulk_apply_offset": 0,
            },
            "16-byte aligned",
        ),
    )

    for inputs, message in cases:
        runtime_inputs = {"source": np.zeros(256, dtype=np.uint8), **inputs}
        with pytest.raises(numsim.NumSimExecutionError, match=message):
            numsim.Engine().run(module, runtime_inputs)


def test_raw_gather4_descriptor_reaches_the_cache_hint_abi(tmp_path):
    module = numsim.transpile(raw_gather4_cache_hint, cache_dir=tmp_path)

    assert "variant::TensorApplyPriorityGather4" in module.rust_source
    assert "variant::TensorPrefetchEvictLastGather4" in module.rust_source


def test_tensor_map_selector_accepts_independent_lane_hint_descriptors(tmp_path):
    module = numsim.transpile(varying_typed_tensor_map_hint, cache_dir=tmp_path)
    inputs = {
        "first_map": _tensor_map(np.zeros((4, 4), dtype=np.float32)),
        "second_map": _tensor_map(np.ones((4, 4), dtype=np.float32)),
        "single_lane": 0,
    }

    # Descriptor selection is now lowered per issuing lane; independent cache
    # hints do not require all lanes to select the same descriptor.
    for single_lane in (0, 1):
        inputs["single_lane"] = single_lane
        result = numsim.Engine().run(module, inputs)
        assert result.verdict == "clean"
