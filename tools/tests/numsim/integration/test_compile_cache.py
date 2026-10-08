from __future__ import annotations

import json
from dataclasses import replace

import pytest
from tvm import tirx
from tvm.ir import load_json, save_json
from tvm_ffi import Object

from tests.numsim.support.kernels import lane_add
from tirx_harness.numsim import UnsupportedTIRxError
from tirx_harness.numsim.transpiler import compile_cache, suspend_scaffold
from tirx_harness.numsim.transpiler.artifact_template import emit_rust_module
from tirx_harness.numsim.transpiler.build import generated_source_identity
from tirx_harness.numsim.transpiler.frontend import (
    analyze,
    attach_source_nodes,
    module_spec_from_manifest,
    verify,
)
from tirx_harness.numsim.transpiler.native_frontend import post_order_nodes
from tirx_harness.numsim.transpiler.source_map import flatten_source_span

_PLAIN = {"analysis_capable": False, "analysis_checker": None}


def _isolated_cache(monkeypatch, tmp_path) -> list[dict]:
    """Serve the compile cache from ``tmp_path`` and record every native compile."""

    monkeypatch.setattr(compile_cache, "default_cache_root", lambda: tmp_path)
    compiles: list[dict] = []
    native = compile_cache.compile_native_module

    def recorded(funcs, **modes):
        compiles.append(modes)
        return native(funcs, **modes)

    monkeypatch.setattr(compile_cache, "compile_native_module", recorded)
    return compiles


def _relocate_source_spans(func, source_name: str, line_delta: int):
    payload = json.loads(save_json(func))
    for node in payload["nodes"]:
        if node["type"] == "ir.SourceName":
            node["data"] = source_name
        elif node["type"] == "ir.Span":
            node["data"]["line"] += line_delta
            node["data"]["end_line"] += line_delta
    return load_json(json.dumps(payload))


def test_source_nodes_preserve_distinct_identities_and_shared_children():
    x = tirx.Var("x", "int32")
    shared = x + 1
    distinct = x + 1
    body = tirx.SeqStmt([tirx.Evaluate(shared), tirx.Evaluate(shared + distinct)])
    nodes = post_order_nodes(body)
    ids = [Object.__hash__(node) for node in nodes]
    assert len(ids) == len(set(ids))
    assert sum(node.same_as(shared) for node in nodes) == 1
    assert sum(node.same_as(distinct) for node in nodes) == 1
    assert ids.index(Object.__hash__(x)) < ids.index(Object.__hash__(shared))
    assert nodes[-1].same_as(body)


def test_one_compile_serves_the_manifest_and_the_generated_source(monkeypatch, tmp_path):
    compiles = _isolated_cache(monkeypatch, tmp_path)

    first, first_source = compile_cache.compile_module_cached(lane_add, **_PLAIN)
    second, second_source = compile_cache.compile_module_cached(lane_add, **_PLAIN)

    assert compiles == [_PLAIN]
    assert first == second
    assert first_source == second_source == emit_rust_module(first, lane_add)
    assert all(entry.node is not None for entry in second.kernels[0].source_map)


def test_cached_compile_rebinds_the_current_source_spans(monkeypatch, tmp_path):
    compiles = _isolated_cache(monkeypatch, tmp_path)

    first, first_source = compile_cache.compile_module_cached(lane_add, **_PLAIN)
    relocated = _relocate_source_spans(lane_add, "/tmp/relocated/lane_add.py", 1000)
    assert generated_source_identity(lane_add) == generated_source_identity(relocated)
    second, second_source = compile_cache.compile_module_cached(relocated, **_PLAIN)

    assert len(compiles) == 1
    assert second_source == first_source
    assert second.to_manifest(include_source_spans=False) == first.to_manifest(
        include_source_spans=False
    )
    leaves = [
        leaf for entry in second.kernels[0].source_map for leaf in flatten_source_span(entry.span)
    ]
    assert leaves
    assert {leaf.source_name for leaf in leaves} == {"/tmp/relocated/lane_add.py"}
    assert min(leaf.line for leaf in leaves) >= 1001


@pytest.mark.parametrize("changed_field", ["source_kind", "unsupported"])
def test_cached_source_attachment_preserves_changed_metadata(changed_field):
    spec = module_spec_from_manifest(analyze(lane_add).to_manifest())
    identity = generated_source_identity(lane_add)
    attached = attach_source_nodes(spec, lane_add, _cache_key=identity)
    verify(attached)
    assert all(entry.node is not None for entry in attached.kernels[0].source_map)
    restored = module_spec_from_manifest(spec.to_manifest())
    assert attach_source_nodes(restored, lane_add, _cache_key=identity) is attached

    kernel = spec.kernels[0]
    if changed_field == "source_kind":
        kernel = replace(
            kernel,
            source_map=(
                replace(kernel.source_map[0], kind="WrongSourceKind"),
                *kernel.source_map[1:],
            ),
        )
        changed = replace(spec, kernels=(kernel,))
        with pytest.raises(ValueError, match="source kind does not match"):
            attach_source_nodes(changed, lane_add, _cache_key=identity)
    else:
        kernel = replace(kernel, unsupported=("op#0:changed_rejection",))
        changed = attach_source_nodes(
            replace(spec, kernels=(kernel,)), lane_add, _cache_key=identity
        )
        with pytest.raises(UnsupportedTIRxError, match="changed_rejection"):
            verify(changed)


def test_corrupt_manifest_entry_is_recompiled(monkeypatch, tmp_path):
    compiles = _isolated_cache(monkeypatch, tmp_path)

    compile_cache.compile_module_cached(lane_add, **_PLAIN)
    next((tmp_path / "compile" / "entries").glob("*/spec.json")).write_text("corrupt")
    restored, source = compile_cache.compile_module_cached(lane_add, **_PLAIN)

    assert len(compiles) == 2
    assert restored.to_manifest(include_source_spans=False) == analyze(lane_add).to_manifest(
        include_source_spans=False
    )
    assert source == emit_rust_module(restored, lane_add)


def test_corrupt_generated_source_entry_is_recompiled(monkeypatch, tmp_path):
    compiles = _isolated_cache(monkeypatch, tmp_path)

    compile_cache.compile_module_cached(lane_add, **_PLAIN)
    next((tmp_path / "compile" / "entries").glob("*/module.rs")).write_text("corrupt")
    restored, source = compile_cache.compile_module_cached(lane_add, **_PLAIN)

    assert len(compiles) == 2
    assert source == emit_rust_module(restored, lane_add)


def test_each_emission_mode_has_its_own_entry(monkeypatch, tmp_path):
    compiles = _isolated_cache(monkeypatch, tmp_path)
    modes = (
        _PLAIN,
        {"analysis_capable": True, "analysis_checker": "synccheck"},
        {"analysis_capable": True, "analysis_checker": "racecheck"},
    )

    sources = [compile_cache.compile_module_cached(lane_add, **mode)[1] for mode in modes]
    repeated = [compile_cache.compile_module_cached(lane_add, **mode)[1] for mode in modes]

    assert len(compiles) == len(modes)
    assert repeated == sources
    assert len(set(sources)) == len(modes)


def test_split_configuration_and_abi_version_have_their_own_entries(monkeypatch, tmp_path):
    compiles = _isolated_cache(monkeypatch, tmp_path)

    compile_cache.compile_module_cached(lane_add, **_PLAIN)
    monkeypatch.setattr(
        suspend_scaffold,
        "_ROOT_SYNC_SPLIT_TARGET_LINES",
        suspend_scaffold._ROOT_SYNC_SPLIT_TARGET_LINES + 1,
    )
    compile_cache.compile_module_cached(lane_add, **_PLAIN)
    monkeypatch.setattr(
        compile_cache, "NUMSIM_ABI_VERSION", compile_cache.NUMSIM_ABI_VERSION + 1
    )
    compile_cache.compile_module_cached(lane_add, **_PLAIN)

    assert len(compiles) == 3


def test_analysis_inner_split_target_is_part_of_the_compile_identity(monkeypatch):
    before = compile_cache.split_configuration_identity()
    monkeypatch.setattr(
        suspend_scaffold,
        "_ANALYSIS_INNER_ASYNC_SPLIT_TARGET_LINES",
        suspend_scaffold._ANALYSIS_INNER_ASYNC_SPLIT_TARGET_LINES + 1,
    )

    after = compile_cache.split_configuration_identity()

    assert before["_ANALYSIS_INNER_ASYNC_SPLIT_TARGET_LINES"] == 64
    assert after["_ANALYSIS_INNER_ASYNC_SPLIT_TARGET_LINES"] == 65
    assert before != after


@pytest.mark.parametrize("invalid", ["missing", "unknown"])
def test_split_configuration_rejects_missing_or_unknown_fields(monkeypatch, invalid):
    from tirx_harness.numsim.transpiler import native_frontend

    thresholds = native_frontend._split_thresholds()
    if invalid == "missing":
        thresholds.pop("_ROOT_SYNC_SPLIT_TARGET_LINES")
    else:
        thresholds["_UNKNOWN_SPLIT_THRESHOLD"] = 1
    with monkeypatch.context() as patch:
        patch.setattr(native_frontend, "_split_thresholds", lambda: thresholds)
        with pytest.raises(ValueError, match=f"{invalid} split threshold"):
            emit_rust_module(analyze(lane_add), lane_add)
    assert emit_rust_module(analyze(lane_add), lane_add)
