#!/usr/bin/env python3
"""FlashInfer ``trtllm_fp8_block_scale_moe`` baseline for DeepSeek-routed MoE."""

from __future__ import annotations

import torch
from flashinfer.fused_moe import trtllm_fp8_block_scale_moe

NUM_EXPERTS = 256
TOP_K = 8
N_GROUP = 8
TOPK_GROUP = 4
INTERMEDIATE_SIZE = 2048


@torch.no_grad()
def run(
    routing_logits,
    routing_bias,
    hidden_states,
    hidden_states_scale,
    gemm1_weights,
    gemm1_weights_scale,
    gemm2_weights,
    gemm2_weights_scale,
    local_expert_offset,
    routed_scaling_factor,
):
    """Call FlashInfer's trtllm FP8 block-scale MoE kernel directly."""

    return trtllm_fp8_block_scale_moe(
        routing_logits=routing_logits,
        routing_bias=routing_bias,
        hidden_states=hidden_states,
        hidden_states_scale=hidden_states_scale,
        gemm1_weights=gemm1_weights,
        gemm1_weights_scale=gemm1_weights_scale,
        gemm2_weights=gemm2_weights,
        gemm2_weights_scale=gemm2_weights_scale,
        num_experts=NUM_EXPERTS,
        top_k=TOP_K,
        n_group=N_GROUP,
        topk_group=TOPK_GROUP,
        intermediate_size=INTERMEDIATE_SIZE,
        local_expert_offset=int(local_expert_offset),
        local_num_experts=gemm1_weights.shape[0],
        routed_scaling_factor=float(routed_scaling_factor),
        routing_method_type=2,
        use_shuffled_weight=False,
        tune_max_num_tokens=max(32768, int(hidden_states.shape[0])),
    )
