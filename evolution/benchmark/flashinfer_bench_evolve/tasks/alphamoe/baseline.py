#!/usr/bin/env python3
"""FlashInfer TRT-LLM FP8 block-scale expert baseline for AlphaMoE.

The public boundary is BF16 activations and preselected int32/FP32 routes.
Input quantization, route processing, expert computation and reduction are
timed. Static weight padding, gate/up conversion and autotuning happen in prepare().
"""

from __future__ import annotations

import torch


@torch.compile(
    fullgraph=True,
    dynamic=True,
    options={"eager_numerics.division_rounding": True},
)
def _quantize(hidden_states):
    # Preserve eager division rounding at FP8 bin boundaries. The compiler
    # fuses this preprocessing; no kernel DSL or additional library is needed.
    rows, columns = hidden_states.shape
    groups = hidden_states.float().reshape(rows, columns // 128, 128)
    scales = groups.abs().amax(dim=-1).clamp_min(1.0e-8) / 448.0
    quantized = (groups / scales.unsqueeze(-1)).clamp(-448.0, 448.0)
    return quantized.to(torch.float8_e4m3fn).reshape(rows, columns), scales


def quantize_inputs(hidden_states):
    """Return FP8 [M, K] values and FP32 [M, K/128] scales.

    Candidates with FP8 entry points should call this inside their timed call.
    Compilation happens during warmup, before CUDA graph capture.
    """
    assert hidden_states.is_cuda and hidden_states.dtype == torch.bfloat16
    assert hidden_states.ndim == 2 and hidden_states.is_contiguous()
    assert hidden_states.shape[1] % 128 == 0
    return _quantize(hidden_states)


def _prepare_weights(contract_args):
    """Adapt expert alignment and logical [gate; up] to TRT-LLM's [up; gate]."""
    args = list(contract_args)
    for index in (3, 4):
        gate, up = args[index].chunk(2, dim=1)
        args[index] = torch.cat((up, gate), dim=1).contiguous()
    # TRT-LLM requires multiples of four experts; the supplied routes stay valid.
    padding = -args[3].shape[0] % 4
    if padding:
        for index in range(3, 7):
            tensor = args[index]
            extra = tensor.new_full(
                (padding, *tensor.shape[1:]),
                1 if index in (4, 6) else 0,
            )
            args[index] = torch.cat((tensor, extra), dim=0)
    return tuple(args)


def _launch(
    hidden_states,
    topk_ids,
    topk_weights,
    gemm1_weights,
    gemm1_weights_scale,
    gemm2_weights,
    gemm2_weights_scale,
    top_k,
    routed_scaling_factor,
):
    from flashinfer.fused_moe import trtllm_fp8_block_scale_routed_moe

    x_q, x_scale = quantize_inputs(hidden_states)
    assert topk_ids.shape == topk_weights.shape == (hidden_states.shape[0], int(top_k))
    # The precomputed routing path consumes final weights and does not apply
    # routed_scaling_factor. Keep this FP32 multiplication inside capture.
    weights = topk_weights
    if float(routed_scaling_factor) != 1.0:
        weights = weights * float(routed_scaling_factor)
    return trtllm_fp8_block_scale_routed_moe(
        topk_ids=(topk_ids, weights),
        routing_bias=None,
        hidden_states=x_q,
        hidden_states_scale=x_scale.t().contiguous(),
        gemm1_weights=gemm1_weights,
        gemm1_weights_scale=gemm1_weights_scale,
        gemm2_weights=gemm2_weights,
        gemm2_weights_scale=gemm2_weights_scale,
        num_experts=gemm1_weights.shape[0],
        top_k=int(top_k),
        n_group=None,
        topk_group=None,
        intermediate_size=gemm2_weights.shape[2],
        local_expert_offset=0,
        local_num_experts=gemm1_weights.shape[0],
        routed_scaling_factor=1.0,
        routing_method_type=5,  # TopK; routes are already selected and weighted.
        use_shuffled_weight=False,
        tune_max_num_tokens=max(1, hidden_states.shape[0]),
    )


@torch.no_grad()
def run(
    hidden_states,
    topk_ids,
    topk_weights,
    gemm1_weights,
    gemm1_weights_scale,
    gemm2_weights,
    gemm2_weights_scale,
    top_k,
    routed_scaling_factor,
):
    """Run the public TRT-LLM operator, preserving all caller inputs."""
    return _launch(
        *_prepare_weights(
            (
                hidden_states,
                topk_ids,
                topk_weights,
                gemm1_weights,
                gemm1_weights_scale,
                gemm2_weights,
                gemm2_weights_scale,
                top_k,
                routed_scaling_factor,
            )
        )
    )


@torch.no_grad()
def prepare(*contract_args):
    """Prepare weights and tune, then capture the complete expert operator."""
    from flashinfer import autotune

    args = _prepare_weights(contract_args)
    with autotune(tuning_buckets=(max(1, args[0].shape[0]),)):
        _launch(*args)
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        _launch(*args)
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = _launch(*args)
    return graph, output, args


@torch.no_grad()
def run_prepared(graph, output, args):
    """Replay quantization, routing and expert work on the retained inputs."""
    graph.replay()
    return output
