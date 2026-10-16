"""Source-level contracts of the baseline's per-shard GEMM and auxiliary kernels."""

import ast

import pytest

from ported.cutlass.sm100_all_gather_gemm import cutlass_gemm, release_flag, sync_nvl_devices


def modifiers(function, instruction):
    return [
        tuple(argument.value for argument in node.args
              if isinstance(argument, ast.Constant) and isinstance(argument.value, str))
        for node in ast.walk(ast.parse(function.script()))
        if isinstance(node, ast.Call) and ast.unparse(node.func) == f"T.ptx.{instruction}"
    ]


@pytest.mark.parametrize("gated", (False, True))
@pytest.mark.parametrize("paired", (False, True))
def test_faithful_shard_does_not_add_a_global_proxy_fence(gated, paired):
    function = cutlass_gemm(
        128, 128, 128, "bf16", "f32", use_2cta=paired,
        mma_tiler=(256, 128) if paired else (128, 64),
        cluster=(2, 1) if paired else (1, 1), use_tma_store=True,
        max_active_clusters=4, gated_a_load=gated,
    ).func

    assert ("proxy", "async", "global", "") not in modifiers(function, "fence")
    waits = [operation for operation in modifiers(function, "cp")
             if operation[:3] == ("async", "bulk", "wait_group")]
    assert waits and all(operation[3] == "read" for operation in waits)


def test_faithful_auxiliary_memory_orders_match_cutlass():
    rendezvous = sync_nvl_devices(4)
    release = release_flag(4)

    atomics = modifiers(rendezvous, "atom")
    loads = modifiers(rendezvous, "ld")
    assert len(atomics) == 2
    assert all(operation[:2] == ("relaxed", "sys") for operation in atomics)
    assert len(loads) == 2
    assert all(operation[1:3] == ("relaxed", "sys") for operation in loads)
    assert modifiers(release, "red")[0][:2] == ("release", "sys")
