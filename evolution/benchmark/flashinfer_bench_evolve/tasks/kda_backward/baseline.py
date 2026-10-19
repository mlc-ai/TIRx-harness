#!/usr/bin/env python3
"""FLA ``chunk_kda`` forward + autograd backward baseline (PR #4636's FLA full-DAG arm)."""

from __future__ import annotations

import os

import torch

# flashinfer PR #4636's bench_recurrent_kda_training.py runs FLA's chunked
# path at chunk size 32 with the fused CUTLASS route disabled.
CHUNK_SIZE = 32
LOWER_BOUND = -5.0


def _kernel_args(
    q, k, v, g, beta, A_log, dt_bias, scale, initial_state, cu_seqlens, do, dfinal_state
):
    """Autograd leaves plus the host copy of ``cu_seqlens`` FLA wants outside timing."""

    leaves = tuple(
        t.detach().requires_grad_(True) for t in (q, k, v, g, beta, A_log, dt_bias, initial_state)
    )
    cu_seqlens_cpu = None if cu_seqlens is None else cu_seqlens.detach().cpu()
    return leaves, float(scale), cu_seqlens, cu_seqlens_cpu, do, dfinal_state


def _launch(leaves, scale, cu_seqlens, cu_seqlens_cpu, do, dfinal_state):
    """FLA's full training DAG: chunk_kda forward, then autograd.grad of both outputs."""

    os.environ["FLA_FLASH_KDA"] = "0"
    os.environ["FLA_TILELANG"] = "0"
    from fla.ops.kda import chunk_kda

    q, k, v, g, beta, A_log, dt_bias, initial_state = leaves
    with torch.enable_grad():
        output, final_state = chunk_kda(
            q,
            k,
            v,
            g,
            beta,
            scale=scale,
            initial_state=initial_state,
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
            use_gate_in_kernel=True,
            use_beta_sigmoid_in_kernel=True,
            safe_gate=True,
            lower_bound=LOWER_BOUND,
            state_v_first=True,
            cu_seqlens=cu_seqlens,
            cu_seqlens_cpu=cu_seqlens_cpu,
            A_log=A_log,
            dt_bias=dt_bias,
            chunk_size=CHUNK_SIZE,
        )
        grads = torch.autograd.grad((output, final_state), leaves, grad_outputs=(do, dfinal_state))
    return (output.detach(), final_state.detach(), *grads)


def run(q, k, v, g, beta, A_log, dt_bias, scale, initial_state, cu_seqlens, do, dfinal_state):
    """Run FLA's chunk_kda forward and backward from the raw layer inputs (eager)."""

    return _launch(
        *_kernel_args(
            q, k, v, g, beta, A_log, dt_bias, scale, initial_state, cu_seqlens, do, dfinal_state
        )
    )


def prepare(*contract_args):
    """Capture the eager call in a CUDA graph outside timing.

    The harness times the CUPTI span of one call (first kernel start to last
    kernel end). FLA's forward plus backward is 31-33 Triton launches, and
    eager launching leaves host-dependent gaps between them inside that span:
    on this shared host the ~1 ms portfolio rows read 2.0-2.8x their kernel
    time at load average ~75 while the 8 ms H96 row reads +2%. PR #4636
    measures the same eager span (``bench_gpu_time(enable_cupti=True,
    cold_l2_cache=True, use_cuda_graph=False)``) on its own host; the graph
    replay here is a deliberate deviation that removes the gaps so the timed
    span is the kernels alone and reproducible. It is not the PR's number: the
    PR's FLA figure includes whatever launch gaps its host had.

    The eager call first finishes Triton's autotune and fills FLA's
    identity-keyed ``tensor_cache`` of chunk indices for this ``cu_seqlens``
    object, so the capture neither tunes nor syncs the host. The graph reads
    the leaves through raw device pointers, so the arguments are returned
    alongside it: the harness drops its own reference to the prepared inputs
    as soon as ``prepare`` returns, and a freed input block would be handed to
    the next allocation and replayed as garbage.
    """

    args = _kernel_args(*contract_args)
    _launch(*args)
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        _launch(*args)
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        outputs = _launch(*args)
    return graph, outputs, args


def run_prepared(graph, outputs, args):
    """Kernel-only dispatch: replay the graph captured by :func:`prepare`.

    ``args`` is not read here; it keeps the captured tensors alive for as
    long as the graph is replayed.
    """

    graph.replay()
    return outputs
