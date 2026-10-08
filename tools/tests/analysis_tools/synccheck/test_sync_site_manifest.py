"""Native Synccheck integration coverage."""

from __future__ import annotations

import json
from pathlib import Path

from tests.numsim.support.kernels import ordering_only_control_calls, scoped_syncs
from tirx_harness.numsim.transpiler.frontend import analyze, module_spec_from_manifest
from tirx_harness.numsim.transpiler.source_map import SequentialSourceSpan, flatten_source_span
from tvm.script import tirx as T


def _inline_wait_impl(barrier):
    T.cuda.mbarrier_wait(barrier, 0)


_inline_wait = T.inline(_inline_wait_impl)


@T.prim_func
def inline_wait_kernel():
    T.device_entry()
    barriers = T.alloc_buffer((1,), "uint64", scope="shared")
    _inline_wait(T.address_of(barriers[0]))


def test_call_source_map_identifies_regular_sync_calls() -> None:
    kernel = analyze(scoped_syncs).kernels[0]
    expected = [
        "tirx.cuda.warp_sync",
        "tirx.cuda.warpgroup_sync",
        "tirx.ptx.bar_sync_count",
        "tirx.cuda.cta_sync",
        "tirx.cuda.cluster_sync",
    ]
    sites = [entry for entry in kernel.source_map if entry.op_name in expected]

    assert [site.op_name for site in sites] == expected
    for site in sites:
        assert kernel.source_map[site.op_id] is site
        assert site.kind == "Call"
        assert site.op_name == str(site.node.op.name)
        assert site.text
        assert site.span is not None


def test_call_source_map_includes_ordering_markers() -> None:
    kernel = analyze(ordering_only_control_calls).kernels[0]
    expected = {"tirx.ptx.fence_mbarrier_init", "tirx.ptx.griddepcontrol"}
    markers = {entry.op_name for entry in kernel.source_map if entry.op_name in expected}

    assert markers == expected


def test_call_source_map_preserves_inline_call_site_before_helper_definition() -> None:
    kernel = analyze(inline_wait_kernel).kernels[0]
    site = next(entry for entry in kernel.source_map if entry.op_name == "tirx.cuda.mbarrier_wait")

    assert isinstance(site.span, SequentialSourceSpan)
    leaves = flatten_source_span(site.span)
    assert len(leaves) == 2
    assert leaves[0].source_name == leaves[1].source_name == str(Path(__file__).resolve())
    assert leaves[0].line > leaves[1].line


def test_call_source_map_is_scoped_by_multi_kernel_parent() -> None:
    spec = analyze((scoped_syncs, ordering_only_control_calls))

    assert [kernel.name for kernel in spec.kernels] == [
        "scoped_syncs",
        "ordering_only_control_calls",
    ]
    assert "tirx.cuda.warp_sync" in {entry.op_name for entry in spec.kernels[0].source_map}
    assert "tirx.ptx.mbarrier_init" in {entry.op_name for entry in spec.kernels[1].source_map}
    for kernel in spec.kernels:
        assert [entry.op_id for entry in kernel.source_map] == list(range(len(kernel.source_map)))


def test_call_source_map_is_stable_json_and_roundtrips_without_ir_nodes() -> None:
    first = analyze((scoped_syncs, ordering_only_control_calls))
    second = analyze((scoped_syncs, ordering_only_control_calls))

    manifest = first.to_manifest()
    assert manifest == second.to_manifest()
    restored = module_spec_from_manifest(json.loads(json.dumps(manifest, sort_keys=True)))
    assert restored.to_manifest() == manifest
    for original, cached in zip(first.kernels, restored.kernels, strict=True):
        assert [entry.op_name for entry in original.source_map] == [
            entry.op_name for entry in cached.source_map
        ]
        assert all(entry.node is None for entry in cached.source_map)
        assert "sync_sites" not in cached.to_manifest()
        assert all("numeric_contract" not in entry.to_dict() for entry in cached.source_map)
    assert "0x" not in json.dumps(manifest, sort_keys=True)
