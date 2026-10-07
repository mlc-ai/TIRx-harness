from __future__ import annotations

import numpy as np
import pytest

from tirx_harness import numsim
from tirx_harness.numsim.checker_report import RaceReport
from tirx_harness.numsim.checkers import _run_racecheck as racecheck
from tvm.script import tirx as T
from tvm.tirx.layout import S, TCol, TileLayout, TLane

_TMEM_LAYOUT = TileLayout(S[(128, 4) : (1 @ TLane, 1 @ TCol)])
_TMEM_WIDE_LAYOUT = TileLayout(S[(128, 8) : (1 @ TLane, 1 @ TCol)])


@T.prim_func
def native_pool_alias_provenance(mode: T.int32, output: T.Buffer((1,), "int32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    pool = T.alloc_buffer((1,), "int32", scope="shared")
    A_shared = T.decl_buffer((1,), "int32", data=pool.data, scope="shared")
    B_shared = T.decl_buffer((1,), "int32", data=pool.data, scope="shared")

    if lane == 0:
        A_shared[0] = 1
        if mode == 0:
            B_shared[0] = 2
            output[0] = A_shared[0]
        elif mode == 1:
            B_shared[0] = 3
            output[0] = B_shared[0]
        else:
            output[0] = A_shared[0]
            B_shared[0] = 4
            output[0] = B_shared[0]


@T.prim_func
def native_raw_ptx_pool_alias_provenance(output: T.Buffer((1,), "uint32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    pool = T.alloc_buffer((1,), "uint32", scope="shared")
    A_shared = T.decl_buffer((1,), "uint32", data=pool.data, scope="shared")
    B_shared = T.decl_buffer((1,), "uint32", data=pool.data, scope="shared")
    loaded = T.alloc_local((1,), "uint32")

    if lane == 0:
        T.ptx.st.shared.u32(A_shared.ptr_to([0]), T.uint32(1))
        T.ptx.st.shared.u32(B_shared.ptr_to([0]), T.uint32(2))
        T.ptx.ld.shared.u32(loaded[0], A_shared.ptr_to([0]))
        output[0] = loaded[0]


@T.prim_func
def native_explicit_view_provenance(output: T.Buffer((1,), "int32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    storage = T.alloc_buffer((4,), "int32", scope="shared")
    matrix = storage.view(2, 2)
    transposed = matrix.rearrange("row col -> col row")

    if lane == 0:
        storage[2] = 7
        output[0] = transposed[0, 1]


@T.prim_func
def native_tmem_view_provenance(output: T.Buffer((32,), "uint32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    tmem = T.decl_buffer((128, 4), "uint32", scope="tmem", layout=_TMEM_LAYOUT, allocated_addr=0)
    tmem_view = tmem.rearrange("(group row) col -> (group row) col", group=2)

    tmem[lane, 0] = T.cast(lane + 1, "uint32")
    output[lane] = tmem_view[lane, 0]


@T.prim_func
def native_tmem_partitioned_view_provenance(output: T.Buffer((32,), "uint32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    tmem = T.decl_buffer(
        (128, 8), "uint32", scope="tmem", layout=_TMEM_WIDE_LAYOUT, allocated_addr=0
    )
    tmem_lo = tmem.sub[:, :4]
    _tmem_hi = tmem.sub[:, 4:]
    tmem_view = tmem.rearrange("(group row) col -> (group row) col", group=2)

    tmem_lo[lane, 0] = T.cast(lane + 1, "uint32")
    output[lane] = tmem_view[lane, 0]


@T.prim_func
def native_tmem_full_extent_subview_provenance(output: T.Buffer((32,), "uint32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    tmem = T.decl_buffer((2, 64, 4), "uint32", scope="tmem", layout=_TMEM_LAYOUT, allocated_addr=0)
    tmem_view = tmem.sub[:, :, 0:4]

    tmem[0, lane, 0] = T.cast(lane + 1, "uint32")
    output[lane] = tmem_view[0, lane, 0]


@T.prim_func
def native_tmem_reused_lifetime_provenance(output: T.Buffer((32,), "uint32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    first = T.decl_buffer((128, 4), "uint32", scope="tmem", layout=_TMEM_LAYOUT, allocated_addr=0)
    second = T.decl_buffer((128, 4), "uint32", scope="tmem", layout=_TMEM_LAYOUT, allocated_addr=0)

    first[lane, 0] = T.cast(lane + 1, "uint32")
    second[lane, 0] = T.cast(lane + 2, "uint32")
    output[lane] = first[lane, 0]


@T.prim_func
def native_tcgen_alloc_result_is_warp_visible(output: T.Buffer((32,), "uint32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    address = T.alloc_buffer((1,), "uint32", scope="shared")

    T.ptx.tcgen05.alloc.cta_group__1.sync.aligned.shared__cta.b32(T.address_of(address[0]), 32)
    output[lane] = address[0]
    T.ptx.tcgen05.dealloc.cta_group__1.sync.aligned.b32(address[0], 32)
    T.ptx.tcgen05.relinquish_alloc_permit.cta_group__1.sync.aligned()


def _run(mode: int, cache_dir):
    return racecheck(
        native_pool_alias_provenance,
        inputs={"mode": np.int32(mode), "output": np.zeros(1, dtype=np.int32)},
        cache_dir=cache_dir,
    )


def test_public_native_stale_logical_name_read_is_review(tmp_path):
    module = numsim.transpile(native_pool_alias_provenance, cache_dir=tmp_path)
    direct = numsim.Engine().run_racecheck_phase(
        module,
        {"mode": np.int32(0), "output": np.zeros(1, dtype=np.int32)},
        inspect_accesses=True,
    )
    assert direct.verdict == "review"
    assert direct.findings == []
    assert [item["kind"] for item in direct.advisories] == ["alias_stale_read"]
    direct_payload = direct.to_dict()
    assert direct_payload["access_count"] == 4
    assert direct_payload["accesses_complete"] is True
    logical_accesses = [
        (access["access_kind"], access["logical_buffer"])
        for access in direct_payload["accesses"]
        if access["space"] == "shared"
    ]
    assert logical_accesses == [
        ("write", "A_shared"),
        ("write", "B_shared"),
        ("read", "A_shared"),
    ]

    report = _run(0, tmp_path)

    assert report.verdict == "review"
    assert [(finding.status, finding.kind) for finding in report.findings] == [
        ("review", "alias_stale_read")
    ]
    with pytest.raises(RuntimeError, match="racecheck review"):
        report.require_clean()

    payload = report.to_dict()["native"]
    assert payload["verdict"] == "review"
    assert payload["findings"] == []
    assert payload["incomplete"] == []
    assert "search" not in payload
    assert "counterexample" not in payload
    assert payload["stats"]["available"] is True
    assert payload["access_count"] == direct_payload["access_count"]
    assert payload["accesses_complete"] is False
    assert payload["accesses"] == []
    assert len(payload["advisories"]) == 1
    advisory = payload["advisories"][0]
    assert advisory["kind"] == "alias_stale_read"
    assert advisory["reader_buffer"] == "A_shared"
    assert advisory["writer_buffer"] == "B_shared"
    assert advisory["space"] == "shared"
    assert advisory["overlaps"] == [
        {"allocation_id": advisory["allocation_id"], "byte_offset": 0, "byte_len": 4, "byte_end": 4}
    ]
    for operation_name in ("reader_operation", "writer_operation"):
        source = advisory[operation_name]["source"]
        assert source["source_text"].strip()
        assert source["source_span"] is not None

    rendered = report.format()
    assert "Reader: warp 0" in rendered
    assert "Writer: warp 0" in rendered


def test_raw_ptx_shared_address_retains_logical_alias_owner(tmp_path):
    report = racecheck(
        native_raw_ptx_pool_alias_provenance,
        inputs={"output": np.zeros(1, dtype=np.uint32)},
        cache_dir=tmp_path,
    )

    assert report.verdict == "review"
    native = report.to_dict()["native"]
    assert native["findings"] == []
    assert native["incomplete"] == []
    assert native["execution_error"] is None
    assert len(native["advisories"]) == 1
    advisory = native["advisories"][0]
    assert advisory["kind"] == "alias_stale_read"
    assert advisory["reader_buffer"] == "A_shared"
    assert advisory["writer_buffer"] == "B_shared"
    assert advisory["space"] == "shared"


@pytest.mark.parametrize("mode", [1, 2], ids=["read-writer-name", "disjoint-lifetimes"])
def test_public_native_non_stale_pool_alias_controls_are_clean(mode: int, tmp_path):
    report = _run(mode, tmp_path)

    report.require_clean()
    assert report.verdict == "clean"
    assert report.findings == []
    payload = report.to_dict()["native"]
    assert payload["verdict"] == "clean"
    assert payload["advisories"] == []
    assert payload["incomplete"] == []
    assert payload["access_count"] > 0
    assert payload["accesses_complete"] is False
    assert payload["accesses"] == []


def test_public_native_explicit_view_keeps_one_logical_identity(tmp_path):
    report = racecheck(
        native_explicit_view_provenance,
        inputs={"output": np.zeros(1, dtype=np.int32)},
        cache_dir=tmp_path,
    )

    report.require_clean()
    native = report.to_dict()["native"]
    assert native["findings"] == []
    assert native["advisories"] == []
    assert native["incomplete"] == []


def test_public_native_tmem_view_keeps_one_logical_identity(tmp_path):
    inputs = {"output": np.zeros(32, dtype=np.uint32)}
    numeric_module = numsim.transpile(
        native_tmem_view_provenance,
        cache_dir=tmp_path,
        _analysis_checker=None,
    )
    numeric = numsim.Engine().run(numeric_module, inputs)
    np.testing.assert_array_equal(numeric.outputs["output"], np.arange(1, 33, dtype=np.uint32))

    report = racecheck(native_tmem_view_provenance, inputs=inputs, cache_dir=tmp_path)
    report.require_clean()
    native = report.to_dict()["native"]
    assert native["findings"] == []
    assert native["advisories"] == []
    assert native["incomplete"] == []


def test_public_native_partitioned_tmem_views_keep_one_logical_identity(tmp_path):
    inputs = {"output": np.zeros(32, dtype=np.uint32)}
    numeric_module = numsim.transpile(
        native_tmem_partitioned_view_provenance,
        cache_dir=tmp_path,
        _analysis_checker=None,
    )
    numeric = numsim.Engine().run(numeric_module, inputs)
    np.testing.assert_array_equal(numeric.outputs["output"], np.arange(1, 33, dtype=np.uint32))

    report = racecheck(native_tmem_partitioned_view_provenance, inputs=inputs, cache_dir=tmp_path)
    report.require_clean()
    native = report.to_dict()["native"]
    assert native["findings"] == []
    assert native["advisories"] == []
    assert native["incomplete"] == []


def test_public_native_full_extent_tmem_subview_keeps_one_logical_identity(tmp_path):
    inputs = {"output": np.zeros(32, dtype=np.uint32)}
    numeric_module = numsim.transpile(
        native_tmem_full_extent_subview_provenance,
        cache_dir=tmp_path,
        _analysis_checker=None,
    )
    numeric = numsim.Engine().run(numeric_module, inputs)
    np.testing.assert_array_equal(numeric.outputs["output"], np.arange(1, 33, dtype=np.uint32))

    report = racecheck(
        native_tmem_full_extent_subview_provenance,
        inputs=inputs,
        cache_dir=tmp_path,
    )
    report.require_clean()
    native = report.to_dict()["native"]
    assert native["findings"] == []
    assert native["advisories"] == []
    assert native["incomplete"] == []


def test_public_native_reused_tmem_lifetime_remains_distinct(tmp_path):
    inputs = {"output": np.zeros(32, dtype=np.uint32)}
    report = racecheck(native_tmem_reused_lifetime_provenance, inputs=inputs, cache_dir=tmp_path)

    assert report.verdict == "review"
    native = report.to_dict()["native"]
    assert native["findings"] == []
    assert [item["kind"] for item in native["advisories"]] == ["alias_stale_read"]
    advisory = native["advisories"][0]
    assert advisory["reader_buffer"] == "first"
    assert advisory["writer_buffer"] == "second"


def test_public_native_tcgen_alloc_result_is_visible_to_every_lane(tmp_path):
    inputs = {"output": np.full(32, np.iinfo(np.uint32).max, dtype=np.uint32)}
    numeric_module = numsim.transpile(
        native_tcgen_alloc_result_is_warp_visible,
        cache_dir=tmp_path,
        _analysis_checker=None,
    )
    numeric = numsim.Engine().run(numeric_module, inputs)
    np.testing.assert_array_equal(numeric.outputs["output"], np.zeros(32, dtype=np.uint32))

    report = racecheck(
        native_tcgen_alloc_result_is_warp_visible,
        inputs=inputs,
        cache_dir=tmp_path,
    )

    report.require_clean()
    native = report.to_dict()["native"]
    assert native["findings"] == []
    assert native["advisories"] == []
    assert native["incomplete"] == []


def test_native_advisory_keeps_error_incomplete_review_precedence():
    advisory = {
        "kind": "alias_stale_read",
        "message": "stale alias",
        "reader_buffer": "A_shared",
        "writer_buffer": "B_shared",
        "space": "shared",
        "allocation_id": 1,
        "overlaps": [{"allocation_id": 1, "byte_offset": 0, "byte_len": 4, "byte_end": 4}],
        "reader_operation": {"global_warp_id": 0, "source_op_id": 2},
        "writer_operation": {"global_warp_id": 0, "source_op_id": 1},
        "occurrences": 1,
    }
    base = {
        "phase": {"topology": {"warps_per_cta": 1}},
        "advisories": [advisory],
        "sync": None,
        "accesses": [],
        "access_count": 0,
        "execution_error": None,
    }

    review = RaceReport.from_native({**base, "verdict": "review", "findings": [], "incomplete": []})
    incomplete = RaceReport.from_native(
        {
            **base,
            "verdict": "incomplete",
            "findings": [],
            "incomplete": [{"kind": "analysis_incomplete", "reason": "coverage_limit", "message": "not exhaustive"}],
        }
    )
    error = RaceReport.from_native(
        {
            **base,
            "verdict": "error",
            "findings": [{"kind": "data_race", "access_pair": "write_read", "message": "race"}],
            "incomplete": [{"kind": "analysis_incomplete", "reason": "coverage_limit", "message": "not exhaustive"}],
        }
    )

    assert review.verdict == "review"
    assert incomplete.verdict == "incomplete"
    assert error.verdict == "error"
    assert {finding.status for finding in error.findings} == {"error", "incomplete", "review"}
