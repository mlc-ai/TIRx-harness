"""Keep baseline instructions and work assignment distinct from semantic repairs."""

import ast
import inspect
from dataclasses import asdict

import pytest
import tvm
import tvm_ffi

from ported.deepgemm._sm100_fp8_fp4_mega_moe.kernel import get_kernel
from tests.numsim.support.peer_mega_moe import Shape, kernel


@pytest.fixture(scope="module", params=[(world, mode) for world in (1, 4)
                                       for mode in ("origin", "sys_fenced")])
def emitted(request):
    world, mode = request.param
    shape = Shape(world=world, num_tokens=(4,) * world, hidden=7168, intermediate_hidden=2048)
    return mode, ast.parse(kernel(shape, ordering=mode).script())


def calls(tree, name):
    return [node for node in ast.walk(tree) if isinstance(node, ast.Call)
            and ast.unparse(node.func) == name]


def modifiers(call):
    return tuple(argument.value for argument in call.args
                 if isinstance(argument, ast.Constant) and isinstance(argument.value, str))


def test_default_port_retains_timeouts_and_reference_memory_forms(emitted):
    mode, tree = emitted
    assert inspect.signature(get_kernel).parameters["ordering"].default == "origin"
    assert bool(calls(tree, "T.cuda.clock64")) == (mode == "origin")
    assert bool(calls(tree, "T.cuda.printf")) == (mode == "origin")
    assert any("nc" in modifiers(call) for call in calls(tree, "T.ptx.ld"))
    if mode == "origin":
        assert any("volatile" in modifiers(call) for call in calls(tree, "T.ptx.ld"))
        assert not calls(tree, "T.cuda.wait_until")
    global_fences = [call for call in calls(tree, "T.ptx.fence")
                     if modifiers(call) == ("proxy", "async", "global", "")]
    assert bool(global_fences) == (mode == "sys_fenced")


def test_combine_keeps_chunks_serial_within_each_tokens_warp(emitted):
    _, tree = emitted
    additions = [call for call in calls(tree, "T.ptx.add") if "bf16" in modifiers(call)]
    assert len(additions) == 2
    enclosing = [node for node in ast.walk(tree) if isinstance(node, ast.For)
                 and all(call in list(ast.walk(node)) for call in additions)]
    assert any(ast.unparse(loop.iter) == "range(2)" for loop in enclosing)


def test_combine_load_sync_is_guarded_by_the_nonempty_mask(emitted):
    _, tree = emitted
    additions = [call for call in calls(tree, "T.ptx.add") if "bf16" in modifiers(call)]
    chunk_loop = next(node for node in ast.walk(tree) if isinstance(node, ast.For)
                      and ast.unparse(node.iter) == "range(2)"
                      and all(call in list(ast.walk(node)) for call in additions))
    copies = calls(chunk_loop, "T.ptx.cp")
    loads = [call for call in copies if "mbarrier::complete_tx::bytes" in modifiers(call)]
    assert len(loads) == 2
    for load in loads:
        guards = [node for node in ast.walk(chunk_loop) if isinstance(node, ast.If)
                  and load in list(ast.walk(node))
                  and not calls(node.test, "T.cuda.elect_sync")]
        assert len(guards) == 1
        assert calls(guards[0], "T.cuda.warp_sync")


def test_peer_offsets_are_selected_from_parameters_without_a_shared_cache(emitted):
    _, tree = emitted
    assert "smem_symm_rank_bases" not in ast.unparse(tree)
    parameters = {argument.arg for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
                  for argument in node.args.args}
    assert "symm_rank_offsets" in parameters
    assert not any(name.startswith("symm_rank_offset_") for name in parameters)
    definitions = [argument for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
                   for argument in node.args.args if argument.arg == "symm_rank_offsets"]
    assert len(definitions) == 1
    annotation = definitions[0].annotation
    assert ast.unparse(annotation.args[0]) == "(72,)"
    assert any(keyword.arg == "scope" and keyword.value.value == "local.param"
               for keyword in annotation.keywords)


def test_unknown_ordering_is_rejected():
    with pytest.raises(ValueError, match="unsupported Mega MoE ordering"):
        kernel(ordering="unknown")


@pytest.mark.parametrize("fast_math", [0, 1])
@pytest.mark.parametrize("activation_clamp", [10.0, float("inf")])
def test_activation_retains_reference_compile_time_branches(fast_math, activation_clamp):
    shape = Shape(world=1, num_tokens=(4,), hidden=1024, intermediate_hidden=512)
    configuration = asdict(shape.config())
    configuration.update(fast_math=fast_math, activation_clamp=activation_clamp)
    tree = ast.parse(get_kernel(**configuration).script())
    divisions = [node for node in ast.walk(tree) if isinstance(node, ast.BinOp)
                 and isinstance(node.op, ast.Div)]
    reciprocals = calls(tree, "T.ptx.rcp")
    assert bool(divisions) == (fast_math == 0)
    assert bool(reciprocals) == (fast_math == 1)
    clamps = calls(tree, "T.cuda.hmin2") + calls(tree, "T.cuda.hmax2")
    assert bool(clamps) == (activation_clamp != float("inf"))


@pytest.mark.parametrize("world,ordering", [(world, mode) for world in (1, 4)
                                          for mode in ("origin", "sys_fenced")])
def test_remote_barrier_addresses_retain_the_references_shared_u32_forms(world, ordering):
    shape = Shape(world=world, num_tokens=(4,) * world, hidden=7168, intermediate_hidden=2048)
    function = kernel(shape, ordering=ordering)
    instructions = []
    tvm_ffi.structural_walk(
        function.body,
        (tvm.ir.Call, lambda call: instructions.append(call)
         if isinstance(call.op, tvm.ir.Op) else None),
    )
    mappings = [call for call in instructions if call.op.name.startswith("tirx.ptx.mapa")]
    assert mappings
    assert sum(call.op.name == "tirx.ptx.mapa_u32" for call in mappings) >= 4
    arrivals = [call for call in instructions if "ptx.mbarrier_arrive" in call.op.name]
    qualifiers_by_call = [tuple(str(argument.value) for argument in call.args
                               if isinstance(argument, tvm.ir.StringImm))
                          for call in arrivals]
    assert sum("shared::cluster" in qualifiers for qualifiers in qualifiers_by_call) >= 3
    for qualifiers in qualifiers_by_call:
        assert any(value in {"shared", "shared::cta", "shared::cluster"} for value in qualifiers)
        assert "release.cluster" not in qualifiers
        assert not ("release" in qualifiers and "cluster" in qualifiers)
