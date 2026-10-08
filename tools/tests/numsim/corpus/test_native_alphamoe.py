from __future__ import annotations

import pytest

from tests.numsim.corpus.kernels.native_moe import prepare_native_alphamoe_case
from tests.numsim.microtests.harness import NUMSIM_GPU_MARK, require_numsim_gpu
from tests.numsim.support._tirx_kernels import load_tirx_kernel
from tests.numsim.support.three_way import run_three_way_case


@NUMSIM_GPU_MARK
def test_alphamoe_matches_gpu_and_reference(pytestconfig, tmp_path):
    require_numsim_gpu(pytestconfig)
    report = run_three_way_case(prepare_native_alphamoe_case(), cache_dir=tmp_path)
    report.require_ok()


@NUMSIM_GPU_MARK
def test_native_alphamoe_public_m8_matches_reference(pytestconfig):
    require_numsim_gpu(pytestconfig)
    module = load_tirx_kernel("alphamoe_fp8_blockscale_qwen3next")
    case = module.prepare_data(label="m8_official", num_tokens=8, num_ctas=80, device="cuda")
    launch = module._launcher(case)
    first = case["output"].clone()
    launch()
    actual = case["output"].clone()
    reference, abs_sum = module._torch_reference(case)
    module.check_correctness(
        {"first": first, "actual": actual, "reference": reference, "abs_sum": abs_sum},
        label="m8_official", num_tokens=8,
    )


@NUMSIM_GPU_MARK
@pytest.mark.parametrize("num_tokens", [8, 16, 32])
def test_native_alphamoe_graph_replay_refreshes_tagged_partials(pytestconfig, num_tokens):
    require_numsim_gpu(pytestconfig)
    import torch

    module = load_tirx_kernel("alphamoe_fp8_blockscale_qwen3next")
    case = module.prepare_data(
        label=f"m{num_tokens}_official",
        num_tokens=num_tokens,
        num_ctas=min(148, torch.cuda.get_device_properties(0).multi_processor_count),
        device="cuda",
    )
    launch = module._launcher(case)
    original_hidden = case["hidden_states"].clone()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch()

    for hidden_is_zero in (True, False, True, False):
        if hidden_is_zero:
            case["hidden_states"].zero_()
        else:
            case["hidden_states"].copy_(original_hidden)
        graph.replay()
        torch.cuda.synchronize()
        if hidden_is_zero:
            assert torch.count_nonzero(case["output"]).item() == 0
        else:
            assert torch.count_nonzero(case["output"]).item() > 0
