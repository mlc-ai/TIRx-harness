"""The faithful Mega MoE gate excludes repaired, stale, or incomplete runs."""

from dataclasses import asdict
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from benchmarks.peer.mega_moe import validate_rows
from ported.deepgemm._sm100_fp8_fp4_mega_moe import data


def row(**changes):
    return {"shape": "shape", "source": "current", "comparison_kind": "port_fidelity",
            "ordering": "origin", "max_abs_diff": 0, "stats_max_abs_diff": 0,
            "input_storage": "shared", "ratio": 1.0, **changes}


@pytest.mark.parametrize("changes", (
    {"source": "stale"}, {"ordering": "sys_fenced"}, {"comparison_kind": "historical"},
    {"max_abs_diff": 1}, {"stats_max_abs_diff": 1}, {"ratio": 1.02},
    {"ratio": float("nan")}, {"ratio": 0}, {"input_storage": "distinct"},
    {"input_storage": None},
))
def test_ineligible_measurements_do_not_establish_parity(changes):
    with pytest.raises(RuntimeError, match="Mega MoE fidelity validation failed"):
        validate_rows([row(**changes)], ["shape"], "current")


def test_every_requested_shape_requires_current_evidence():
    validate_rows([row()], ["shape"], "current")
    with pytest.raises(RuntimeError, match="missing"):
        validate_rows([row()], ["shape", "missing"], "current")


@pytest.mark.parametrize("with_references", [False, True])
def test_benchmark_uses_one_input_allocation_for_both_implementations(monkeypatch, with_references):
    import tirx_kernels.runner as runner

    monkeypatch.setattr(runner, "bind_cuda_assignment", Mock())
    monkeypatch.setattr(runner, "validate_current_cuda_assignment", Mock())
    monkeypatch.setattr(runner, "external_references_enabled", lambda: with_references)
    monkeypatch.setattr(data.torch.cuda, "device_count", lambda: 4)
    monkeypatch.setattr(data.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(data.torch.distributed, "is_initialized", lambda: False)
    monkeypatch.setattr(data.torch, "zeros", Mock())
    monkeypatch.setattr(data.torch, "empty", Mock())
    monkeypatch.setattr(data.torch, "set_default_device", Mock())
    monkeypatch.setattr(data, "load_deep_gemm_mega", lambda: (object(), "test"))
    monkeypatch.setattr(data, "_init_dist_on_assigned_device", lambda *args: (0, 4, "group"))
    create_case = Mock(side_effect=lambda *args: SimpleNamespace())
    copy_inputs = Mock()
    cleanup = Mock()
    invocation = Mock(side_effect=RuntimeError("stop before launch"))
    monkeypatch.setattr(data, "create_case", create_case)
    monkeypatch.setattr(data, "_copy_inputs_into_symm_buffer", copy_inputs)
    monkeypatch.setattr(data, "_prepare_tirx_invocation", invocation)
    monkeypatch.setattr(data, "_cleanup_distinct_cases", cleanup)
    monkeypatch.setattr(data, "_destroy_process_group", Mock())
    with pytest.raises(RuntimeError, match="stop before launch"):
        data._run_worker(0, 0, "gpu0", asdict(data.MegaMoeConfig(num_processes=4)), "bench")
    create_case.assert_called_once()
    shared_case = invocation.call_args.args[0]
    copy_inputs.assert_called_once_with(shared_case)
    cleanup.assert_called_once_with(None, shared_case if with_references else None, shared_case)


def test_compile_keeps_the_selected_target_active_for_the_cuda_callback(monkeypatch):
    import tvm
    import tirx_kernels.runner as runner
    from ported.deepgemm._sm100_fp8_fp4_mega_moe import kernel, spec

    target = tvm.target.Target({"kind": "cuda", "arch": "sm_100a"})
    executable = object()
    monkeypatch.setattr(runner, "cuda_target", lambda: target)
    monkeypatch.setattr(kernel, "get_kernel", lambda **kwargs: object())
    monkeypatch.setattr(tvm, "IRModule", lambda functions: functions)

    def compile_module(module, target, tir_pipeline):
        current = tvm.target.Target.current(allow_none=True)
        assert current is not None
        assert current.arch == target.arch
        assert tir_pipeline == "tirx"
        assert data.os.environ["TVM_CUDA_COMPILE_MODE"] == "nvrtc"
        return executable

    monkeypatch.setattr(tvm, "compile", compile_module)
    spec._compile_tirx_mega_moe_for_config.cache_clear()
    try:
        result = spec._compile_tirx_mega_moe_for_config(
            **asdict(data.MegaMoeConfig(num_processes=1)),
            collect_stats=False,
            cuda_compile_mode="nvrtc",
        )
        assert result is executable
    finally:
        spec._compile_tirx_mega_moe_for_config.cache_clear()
    assert tvm.target.Target.current(allow_none=True) is None
