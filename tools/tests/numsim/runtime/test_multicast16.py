"""Explicit 16-bit multicast spellings retain the existing physical protocol."""

import numpy as np
import pytest
from tvm.script import tirx as T
from tvm.ir import Call
from tvm_ffi import structural_map

from tirx_harness import numsim, racecheck, synccheck
from tirx_harness.numsim.transpiler.ptx_dialect import decode_ptx_call
from tests.numsim.runtime.test_non_tensor_bulk_forms import lane_varying_bulk_multicast_cta_mask
from tests.numsim.runtime.test_raw_tma_codegen import (
    _tensor_map,
    raw_tma_multicast_cta_group1_per_target_barriers,
    raw_tma_multicast_cta_group2_across_pairs,
)
from tests.numsim.runtime.test_tcgen_codegen import (
    tcgen_commit_dynamic_single_bit_mask_is_multicast,
)
from tests.numsim.support.kernels import tcgen_commit_runtime_multicast
from tests.numsim.support.manifest import call_op_names


def explicit_multicast(kernel, width=16, *, mask=None, semantics=""):
    """Rebuild only the tested instruction through the public PTX builder."""
    replacements = 0

    def rewrite(node):
        nonlocal replacements
        if not str(getattr(getattr(node, "op", None), "name", "")).startswith("tirx.ptx."):
            return node
        decoded = decode_ptx_call(node)
        if decoded.modifiers.get("multicast") != "multicast::cluster":
            return node
        tokens = [
            f"multicast::cluster::{width}b" if token == "multicast::cluster" else token
            for token in decoded.modifiers.values()
            if token
        ]
        if semantics:
            tokens.insert(2, semantics)
            if semantics.startswith("relaxed."):
                tokens.append("b128")
        family = "tcgen05" if "tcgen05" in decoded.op_name else "cp"
        arguments = []
        for name, values in decoded.operands.items():
            if name in {"mask", "cta_mask"}:
                value = values[0] if mask is None else mask
                arguments.append(T.Cast(f"uint{width}", value))
            else:
                arguments.extend(values)
        replacements += 1
        return T.ptx[".".join([family, *tokens])](*arguments, pred=decoded.predicate)

    result = kernel.with_body(structural_map(kernel.body, (Call, rewrite)))
    assert replacements == 1
    return result


def _check_and_run(kernel, inputs, expected, tmp_path):
    def fresh_inputs():
        return {
            name: value.copy() if isinstance(value, np.ndarray) else value
            for name, value in inputs.items()
        }

    for checker in (synccheck, racecheck):
        checker(kernel, fresh_inputs()).require_clean()
    module = numsim.transpile(kernel, cache_dir=tmp_path)
    result = numsim.Engine().run(module, fresh_inputs())
    np.testing.assert_array_equal(result.outputs["output"], expected)
    assert any(
        "multicast16" in op_name or "multicast32" in op_name or "commit_multicast_width" in op_name
        for op_name in call_op_names(module.spec.kernels[0])
    )


@pytest.mark.parametrize("width", [16, 32])
def test_bulk_multicast16_preserves_lane_resolved_targets(tmp_path, width):
    source = np.arange(32, dtype=np.uint8) ^ np.uint8(0x5A)
    for semantics in ("", "weak", "relaxed.cta", "relaxed.cluster", "relaxed.gpu", "relaxed.sys"):
        _check_and_run(
            explicit_multicast(lane_varying_bulk_multicast_cta_mask, width, semantics=semantics),
            {"source": source, "output": np.zeros((2, 16), np.uint8)},
            source.reshape(2, 16),
            tmp_path,
        )


@pytest.mark.parametrize("ctas", [2, 4])
@pytest.mark.parametrize("width", [16, 32])
def test_tensor_multicast16_completes_each_target(tmp_path, ctas, width):
    source = np.arange(4, dtype=np.float32) + np.float32(0.25)
    input_map, _ = _tensor_map(source, global_shape=(4,), global_strides=(), box_shape=(4,))
    original = (
        raw_tma_multicast_cta_group1_per_target_barriers
        if ctas == 2
        else raw_tma_multicast_cta_group2_across_pairs
    )
    _check_and_run(
        explicit_multicast(original, width),
        {"input_map": input_map, "output": np.zeros((ctas, 4), np.float32)},
        np.broadcast_to(source, (ctas, 4)),
        tmp_path,
    )


@pytest.mark.parametrize("group", [1, 2])
@pytest.mark.parametrize("width", [16, 32])
def test_tcgen_multicast16_preserves_commit_targets(tmp_path, group, width):
    original = (
        tcgen_commit_dynamic_single_bit_mask_is_multicast
        if group == 1
        else tcgen_commit_runtime_multicast
    )
    inputs = {"output": np.zeros(group, np.uint32)}
    if group == 1:
        inputs["cta_mask"] = 2
    _check_and_run(explicit_multicast(original, width), inputs, np.ones(group, np.uint32), tmp_path)


@pytest.mark.parametrize(
    "kernel",
    [
        lane_varying_bulk_multicast_cta_mask,
        raw_tma_multicast_cta_group1_per_target_barriers,
        tcgen_commit_dynamic_single_bit_mask_is_multicast,
    ],
)
@pytest.mark.parametrize("width", [16, 32])
def test_multicast16_rejects_target_outside_cluster(kernel, width):
    source = np.arange(4, dtype=np.float32)
    input_map, _ = _tensor_map(source, global_shape=(4,), global_strides=(), box_shape=(4,))
    inputs = {"source": np.zeros(32, np.uint8), "output": np.zeros((2, 16), np.uint8)}
    if kernel is raw_tma_multicast_cta_group1_per_target_barriers:
        inputs = {"input_map": input_map, "output": np.zeros((2, 4), np.float32)}
    elif kernel is tcgen_commit_dynamic_single_bit_mask_is_multicast:
        inputs = {"cta_mask": 2, "output": np.zeros(1, np.uint32)}
    for checker in (synccheck, racecheck):
        report = checker(explicit_multicast(kernel, width, mask=4), inputs)
        assert report.verdict == "error"
        assert "outside cluster" in str(report.to_dict())
