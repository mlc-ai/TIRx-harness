#!/usr/bin/env python3
"""FLA chunk_kda baseline for Kimi K3 KDA forward.

Returns both scored operator outputs: the bf16 output and the fp32 final
recurrent state. ``output_final_state=True`` makes FLA allocate the fp32
``[num_seqs, H, 128, 128]`` state buffer and hand it to FlashKDA as the
kernel's state-writeback destination, which selects the state-storing
kernel instantiation, so this baseline pays for the state epilogue that
the task requires of every candidate.
"""

from __future__ import annotations

import os

import torch

# Use FlashKDA's fused CUTLASS inference backend through the public FLA wrapper.
os.environ["FLA_FLASH_KDA"] = "1"
os.environ["FLA_TILELANG"] = "0"

from fla.ops.kda import chunk_kda


@torch.no_grad()
def run(q, k, v, g, beta, A_log, dt_bias, scale, initial_state, cu_seqlens=None):
    # The standalone harness evaluates the native reference in the same process,
    # so restore this solution's measured backend choice for every invocation.
    os.environ["FLA_FLASH_KDA"] = "1"
    os.environ["FLA_TILELANG"] = "0"
    output, final_state = chunk_kda(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=float(scale),
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=True,
        use_beta_sigmoid_in_kernel=True,
        state_v_first=True,
        safe_gate=True,
        lower_bound=-5.0,
        A_log=A_log,
        dt_bias=dt_bias,
        initial_state=initial_state,
        output_final_state=True,
        cu_seqlens=cu_seqlens,
    )
    return output, final_state
