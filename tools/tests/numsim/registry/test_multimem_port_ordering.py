"""Check the memory-ordering contract of the public full-kernel builder."""

from __future__ import annotations

import ast

import pytest

from ported.cutlass.sm100_all_reduce import two_shot_all_reduce
from ported.cutlass.sm100_gemm_all_reduce import gemm_all_reduce
from tirx_harness.numsim.transpiler.host_prelude import normalize_host_tensor_map_prelude


def _kernel(protocol, use_tma_store, use_2cta, rank, **options):
    return gemm_all_reduce(
        512,
        128,
        128,
        "bf16",
        "f32",
        rank,
        4,
        use_2cta=use_2cta,
        mma_tiler=(256, 128) if use_2cta else (128, 64),
        cluster=(2, 1) if use_2cta else (1, 1),
        use_tma_store=use_tma_store,
        protocol=protocol,
        max_active_clusters=1,
        flag_len=16,
        **options,
    ).func


def _modifiers(function, instruction):
    tree = ast.parse(function.script())
    return [
        tuple(
            argument.value
            for argument in node.args
            if isinstance(argument, ast.Constant) and isinstance(argument.value, str)
        )
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and ast.unparse(node.func) == f"T.ptx.{instruction}"
    ]


@pytest.mark.parametrize("protocol", ("cutlass", "flashinfer"))
@pytest.mark.parametrize("use_tma_store", (True, False))
@pytest.mark.parametrize("use_2cta", (True, False))
@pytest.mark.parametrize("rank", (None, 0))
def test_full_kernel_system_publication_is_explicit(protocol, use_tma_store, use_2cta, rank):
    function = _kernel(protocol, use_tma_store, use_2cta, rank, ordering="sys_fenced")
    arrivals = _modifiers(function, "multimem_red")
    polls = _modifiers(function, "atom")

    assert arrivals and all(modifiers[:2] == ("release", "sys") for modifiers in arrivals)
    assert polls and all(modifiers[:2] == ("acquire", "sys") for modifiers in polls)
    assert ("proxy", "alias", "", "") in _modifiers(function, "fence")
    parameters = {parameter.name for parameter in function.params}
    assert "out" in parameters
    assert ("rank" in parameters) == (rank is None)


@pytest.mark.parametrize("protocol", ("cutlass", "flashinfer"))
@pytest.mark.parametrize("use_tma_store", (True, False))
def test_full_kernel_defaults_to_baseline_ordering(protocol, use_tma_store):
    function = _kernel(protocol, use_tma_store, False, None)

    assert any(modifiers[1] == "gpu" for modifiers in _modifiers(function, "multimem_red"))
    assert any(modifiers[1] == "gpu" for modifiers in _modifiers(function, "atom"))
    if protocol == "cutlass":
        assert not any(
            modifiers[:2] == ("acq_rel", "gpu")
            for modifiers in _modifiers(function, "fence")
        )
        assert ("proxy", "alias", "", "") not in _modifiers(function, "fence")


@pytest.mark.parametrize("protocol", ("cutlass", "flashinfer"))
@pytest.mark.parametrize("use_tma_store", (True, False))
def test_baseline_flag_operations_use_elected_leaders(protocol, use_tma_store):
    faithful = _kernel(protocol, use_tma_store, False, None)
    strengthened = _kernel(protocol, use_tma_store, False, None, ordering="sys_fenced")

    assert len(_modifiers(faithful, "elect_sync")) - len(
        _modifiers(strengthened, "elect_sync")
    ) == (4 if use_tma_store else 3)


@pytest.mark.parametrize("protocol", ("cutlass", "flashinfer"))
@pytest.mark.parametrize("use_2cta", (True, False))
def test_baseline_bulk_store_waits_are_not_silently_strengthened(protocol, use_2cta):
    function = _kernel(protocol, True, use_2cta, None)
    waits = [
        modifiers for modifiers in _modifiers(function, "cp")
        if modifiers[:3] == ("async", "bulk", "wait_group")
    ]

    assert waits
    if protocol == "cutlass":
        assert all(modifiers[3] == "read" for modifiers in waits)
    else:
        assert any(modifiers[3] == "" for modifiers in waits)


def test_two_shot_matches_the_baseline_constexpr_loop():
    tree = ast.parse(two_shot_all_reduce(8, 4, 2).script())
    unrolled = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.For) and isinstance(node.iter, ast.Call)
        and ast.unparse(node.iter.func) == "T.unroll"
    ]

    assert len(unrolled) == 1
    assert ast.literal_eval(unrolled[0].iter.args[-1]) == 32


@pytest.mark.parametrize("rank", range(4))
def test_two_shot_preserves_the_baseline_constexpr_rank(rank):
    function = two_shot_all_reduce(8, 4, 2)
    specialized = function.specialize({function.params[0]: rank})
    assert [str(parameter.name) for parameter in specialized.params] == [
        "data_in", "data_out", "in_mc", "out_mc", "flag", "flag_mc",
    ]


def test_unknown_ordering_is_rejected():
    with pytest.raises(ValueError, match="unsupported all-reduce ordering"):
        _kernel("cutlass", True, False, None, ordering="unknown")


@pytest.mark.parametrize("protocol", ("cutlass", "flashinfer"))
@pytest.mark.parametrize("use_tma_store", (True, False))
def test_full_kernel_host_descriptors_normalize_for_both_epilogues(protocol, use_tma_store):
    normalized = normalize_host_tensor_map_prelude(_kernel(protocol, use_tma_store, False, None))

    assert normalized is not None
