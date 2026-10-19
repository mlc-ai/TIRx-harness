"""Standalone benchmark for Kimi K3 Delta Attention forward candidates."""

from __future__ import annotations

import functools
import hashlib
import math
from itertools import pairwise

import torch

from ...benchmark_common import (
    BenchConfig,
    choose_device,
    compare_outputs,
    config_from_env,
    load_candidate_override,
    load_task_reference,
    load_task_workloads,
    run_benchmark,
)

TASK_NAME = "kda_forward"
HEAD_DIM = 128

# Tolerances match KDA-internal's authoritative judge
# (bench_kda_forward_standalone.py). ATOL is scaled by each output's reference
# RMS, so the bf16 output and the order-one fp32 recurrent state are held to the
# same relative standard; MAX_REL_L2 is the normalized RMS error ratio, one
# bound per scored output.
ATOL = 0.5
RTOL = 0.05
REQUIRED_MATCHED_RATIO = 1.0
MAX_REL_L2 = 0.03
REQUIRED_RMS_ERROR_RATIOS = (MAX_REL_L2, MAX_REL_L2)
# Repeated keys with weak decay expose cancellation in chunk formulations.
# Match the judge's relaxed tolerances for these probes.
CORRELATED_TOLERANCE = {
    "atol": 1.0,
    "rtol": 0.2,
    "required_rms_error_ratios": (0.15, 0.15),
}


# Correctness-only probes at the operator's numerical edge, ported verbatim
# from KDA-internal's judge. Their inputs are adversarial rather than
# K3-realistic: ``saturated`` holds every channel at the bounded gate's floor
# (-5 nats, 7.21 bits of decay per token, 461.7 per 64-token chunk); ``strong``
# is the K3 form with ``A_log = 0`` and ``dt_bias = 0`` (about 231 bits per
# chunk); ``mixed_depth`` saturates half the channels of every head; ``switch``
# alternates saturated and realistic 128-token bands; ``boundary`` holds every
# channel at a constant 2.72 bits per token (174 bits per chunk), a moderate
# depth where a factorisation that is exact only for realistic gates already
# fails; ``correlated`` repeats one unit key and query for every token with
# beta ~ 0.95 and no decay, where a bf16 power-series inverse of the
# intra-chunk system loses its cancellation; ``random_depth`` is the input
# recipe of FLA's own KDA tests, which mixes every decay depth from none to
# saturated across the channels of one head. They are never timed and never
# enter a geomean: a candidate must simply match the reference.
# Each case: (num_heads, regime, seq_lens_per_batch_item).
_STRESS_PACKED = [1300, 65, 1, 2048, 682]
STRESS_CASES = [
    (64, "saturated", _STRESS_PACKED),
    (96, "saturated", [4096]),
    (64, "strong", [4096]),
    (96, "strong", _STRESS_PACKED),
    (64, "mixed_depth", [1024] * 4),
    (96, "switch", _STRESS_PACKED),
    (64, "boundary", [64, 4032]),
    (96, "correlated", [65, 321, 3710]),
    (64, "random_depth", _STRESS_PACKED),
    (96, "random_depth", [4096]),
]
# Exact probes: checked against an fp64 token-by-token recurrence of the
# operator definition instead of the FLA reference, so the verdict on these
# regimes does not depend on any production kernel. Short layouts keep the
# recurrence cheap and exercise sequence lengths that are not multiples of the
# 64-token chunk.
EXACT_CASES = [
    (64, "random_depth", [1, 17, 63, 64, 65, 127, 128, 129]),
    (96, "correlated", [65, 321, 96]),
]
_SATURATED_A_LOG = math.log(16.0)
_SATURATED_GATE = 3.0  # with dt_bias = 0: exp(A_log) * g = 48, sigmoid saturated
_SWITCH_BAND = 128  # tokens per band
_SWITCH_GATE = 10.0  # saturates against the realistic dt_bias floor (-6.9)
_BOUNDARY_GATE = -0.5  # with A_log = dt_bias = 0: -5 sigmoid(-0.5) = -2.72 bits/token
_CORRELATED_BETA = 3.0  # sigmoid(3) = 0.953
_CORRELATED_GATE = -3.0  # with A_log = log 16, dt_bias = 0: sigmoid(-48) ~ 0, no decay


def _apply_stress_regime(regime, q, k, g, beta, A_log, dt_bias):
    """Push the inputs of a stress probe into its regime (in place)."""

    heads = A_log.shape[0]
    if regime == "saturated":
        A_log.fill_(_SATURATED_A_LOG)
        dt_bias.zero_()
        g.fill_(_SATURATED_GATE)
    elif regime == "strong":
        A_log.zero_()
        dt_bias.zero_()
    elif regime == "mixed_depth":
        A_log.fill_(_SATURATED_A_LOG)
        deep = torch.rand(heads, HEAD_DIM, device=g.device) < 0.5
        dt_bias.view(heads, HEAD_DIM).masked_fill_(deep, 0.0)
        g.masked_fill_(deep[None, None], _SATURATED_GATE)
    elif regime == "switch":
        A_log.fill_(_SATURATED_A_LOG)
        tokens = torch.arange(g.shape[1], device=g.device)
        deep = (tokens // _SWITCH_BAND) % 2 == 0
        g.masked_fill_(deep[None, :, None, None], _SWITCH_GATE)
    elif regime == "boundary":
        A_log.zero_()
        dt_bias.zero_()
        g.fill_(_BOUNDARY_GATE)
    elif regime == "correlated":
        q.zero_()
        q[..., 0] = 1.0
        k.copy_(q)
        beta.fill_(_CORRELATED_BETA)
        A_log.fill_(_SATURATED_A_LOG)
        dt_bias.zero_()
        g.fill_(_CORRELATED_GATE)
    elif regime == "random_depth":
        A_log.normal_()
        dt_bias.normal_()
        g.copy_(torch.randn(g.shape, dtype=torch.float32, device=g.device))
        beta.copy_(torch.randn(beta.shape, dtype=torch.float32, device=beta.device))
    else:
        raise ValueError(f"unknown stress regime {regime!r}")


def recurrence_forward(q, k, v, g, beta, A_log, dt_bias, scale, initial_state, cu_seqlens=None):
    """The operator definition itself, evaluated token by token in fp64.

    The exact reference for the ``-exact`` probes, independent of every
    production kernel. Returns the bf16 output and the fp32 V-first final state.
    """

    heads = A_log.shape[0]
    q, k = q[0].double(), k[0].double()
    q = q * torch.rsqrt(q.square().sum(-1, keepdim=True) + 1e-6)
    k = k * torch.rsqrt(k.square().sum(-1, keepdim=True) + 1e-6)
    x = A_log.double().exp()[None, :, None] * (
        g[0].double() + dt_bias.double().view(1, heads, HEAD_DIM)
    )
    decay = torch.exp(-5.0 * torch.sigmoid(x))
    weight = beta[0].double().sigmoid()
    output = torch.empty(v.shape[1:], dtype=torch.float64, device=v.device)
    final_state = torch.empty(initial_state.shape, dtype=torch.float64, device=v.device)
    bounds = [0, v.shape[1]] if cu_seqlens is None else cu_seqlens.tolist()
    for seq, (start, end) in enumerate(pairwise(bounds)):
        state = initial_state[seq].double()  # [H, V, K]
        for t in range(start, end):
            state = state * decay[t, :, None, :]
            delta = (v[0, t].double() - (state * k[t, :, None, :]).sum(-1)) * weight[t, :, None]
            state = state + delta[:, :, None] * k[t, :, None, :]
            output[t] = (state * q[t, :, None, :]).sum(-1) * scale
        final_state[seq] = state
    return output[None].to(torch.bfloat16), final_state.to(torch.float32)


def make_stress_workloads(shape=None):
    """Correctness-only probes; ``shape`` pins every probe to one single-sequence
    ``(num_heads, total_tokens)`` so a fixed-shape task keeps its shape."""

    entries = {}
    for exact, cases in ((False, STRESS_CASES), (True, EXACT_CASES)):
        for num_heads, regime, seq_lens in cases:
            if shape is not None:
                num_heads, seq_lens = shape[0], [shape[1]]
            entry_id = f"h{num_heads}-{regime}" + ("-exact" if exact else "")
            if entry_id in entries:
                continue
            entry = {
                "suite": "stress",
                "id": entry_id,
                "axes": {
                    "batch_size": 1,
                    "num_heads": num_heads,
                    "head_dim": HEAD_DIM,
                    "total_tokens": sum(seq_lens),
                    "num_seqs": len(seq_lens),
                },
                "seq_lens": tuple(seq_lens),
                "regime": regime,
                "timed": False,
                "raw": None,
            }
            if exact:
                # Named rather than bound: workload rows are shipped to the
                # benchmark server as JSON, so an entry may only hold plain
                # data. run_suite resolves the name back to the callable.
                entry["reference"] = "recurrence_forward"
            if regime == "correlated":
                entry.update(CORRELATED_TOLERANCE)
            entries[entry_id] = entry
    return list(entries.values())


def default_config() -> BenchConfig:
    """This task's configuration, read from the environment at call time."""

    return config_from_env(warmup=3, iters=30, trials=3)


def make_workloads(config: BenchConfig | None = None):
    """Pair full-sweep shapes with saturated gates; keep the max-shape smoke row.

    A full sweep also carries the correctness-only stress and exact probes, so
    the suite gates on KDA-internal's numerical edge cases as well as its timed
    rows. A max-shape run pins the untimed probes to its selected shape.
    """

    cfg = config or default_config()
    official = []
    if cfg.include_official:
        for row in load_task_workloads(
            TASK_NAME,
            cfg.max_official,
            shape_mode=cfg.shape_mode,
            # All workloads have 8192 tokens. Leaving num_seqs out makes the
            # first H=96 row (the fixed-length case) the max-shape smoke case.
            shape_axes=("total_tokens", "num_heads"),
        ):
            workload = row["workload"]
            official.append(
                {
                    "suite": "official",
                    "id": workload["uuid"],
                    "gate_config": "standard",
                    "axes": workload["axes"],
                    "seq_lens": tuple(workload["inputs"]["cu_seqlens"]["value"]),
                    "raw": workload,
                }
            )
            if cfg.shape_mode.lower() in ("all", "full"):
                official.append(
                    {
                        **official[-1],
                        "id": f"{workload['uuid']}-saturated",
                        "gate_config": "saturated",
                    }
                )
    # The correctness-only probes travel with the suite rather than with a
    # shape sweep: KDA-internal's fixed-shape variant keeps them too, pinned to
    # its own shape so only the gate inputs and values vary.
    if official:
        official.extend(make_stress_workloads(_pinned_shape(official)))
    return official


def _entry_seed(entry):
    # Pair both gate configurations with the original workload's random inputs.
    # Stress probes have no published workload row, so they seed off their id.
    key = entry["raw"]["uuid"] if entry.get("raw") else entry["id"]
    raw = f"kda-forward:{key}:{entry['axes']}".encode()
    return int(hashlib.sha256(raw).hexdigest()[:8], 16)


def _randn(shape, dtype, device, scale=1.0):
    return (torch.randn(shape, dtype=torch.float32, device=device) * scale).to(dtype)


def _gate_parameters(num_heads, head_dim, device):
    A_log = torch.log(
        torch.empty(num_heads, dtype=torch.float32, device=device).uniform_(1.0, 16.0)
    )
    dt = torch.exp(
        torch.rand(
            num_heads * head_dim,
            dtype=torch.float32,
            device=device,
        )
        * (math.log(0.1) - math.log(0.001))
        + math.log(0.001)
    ).clamp_(min=1e-4)
    dt_bias = dt + torch.log(-torch.expm1(-dt))
    return A_log, dt_bias


def make_inputs(entry, device):
    axes = entry["axes"]
    batch_size = axes["batch_size"]
    total_tokens = axes["total_tokens"]
    num_heads = axes["num_heads"]
    head_dim = axes["head_dim"]
    seq_lens = entry["seq_lens"]
    torch.manual_seed(_entry_seed(entry))
    A_log, dt_bias = _gate_parameters(num_heads, head_dim, device)
    cu_seqlens = None
    if len(seq_lens) > 1:
        cu_seqlens = torch.tensor(
            (0, *torch.tensor(seq_lens, dtype=torch.int64).cumsum(0).tolist()),
            dtype=torch.int64,
            device=device,
        )
    inputs = [
        _randn(
            (batch_size, total_tokens, num_heads, head_dim),
            torch.bfloat16,
            device,
            0.5,
        ),
        _randn(
            (batch_size, total_tokens, num_heads, head_dim),
            torch.bfloat16,
            device,
            0.5,
        ),
        _randn(
            (batch_size, total_tokens, num_heads, head_dim),
            torch.bfloat16,
            device,
            0.5,
        ),
        _randn(
            (batch_size, total_tokens, num_heads, head_dim),
            torch.bfloat16,
            device,
            0.5,
        ),
        _randn((batch_size, total_tokens, num_heads), torch.bfloat16, device, 0.5),
        A_log,
        dt_bias,
        float(entry["raw"]["inputs"]["scale"]["value"])
        if entry.get("raw")
        else 1.0 / math.sqrt(head_dim),
        _randn(
            (len(seq_lens), num_heads, head_dim, head_dim),
            torch.float32,
            device,
            0.25,
        ),
        cu_seqlens,
    ]
    if entry.get("gate_config", "standard") == "saturated":
        inputs[3].fill_(3.0)
        A_log.fill_(math.log(16.0))
        dt_bias.zero_()
    if entry.get("regime"):
        q, k, _v, g, beta = inputs[0], inputs[1], inputs[2], inputs[3], inputs[4]
        _apply_stress_regime(entry["regime"], q, k, g, beta, A_log, dt_bias)
    return inputs


def tirx_prepare(
    solution_module,
    q,
    k,
    v,
    g,
    beta,
    a_log,
    dt_bias,
    scale,
    initial_state,
    cu_seqlens,
):
    """Bind fresh gate inputs to a TIRx KDA-forward candidate outside timing."""

    batch_size, total_tokens, num_heads, _ = q.shape
    data = {
        "q": q,
        "k": k,
        "v": v,
        "g": g,
        "beta": beta,
        "A_log": a_log,
        "dt_bias": dt_bias,
        "scale": float(scale),
        "initial_state": initial_state,
        "cu_seqlens": cu_seqlens,
        "output": torch.empty_like(v),
        # Both scored outputs are preallocated here, outside the timed path; the
        # candidate must overwrite every element of each on every invocation.
        "final_state": torch.empty_like(initial_state),
    }
    kernel_fn = solution_module.setup(data, batch_size, total_tokens, num_heads)
    return kernel_fn, data


def tirx_run(kernel_fn, data):
    kernel_fn()
    return data["output"], data["final_state"]


def _pinned_shape(entries):
    """The single ``(num_heads, total_tokens)`` a narrowed suite runs on, if any.

    A shape-pinned task selects one official row; its holdout probes then keep
    that shape instead of sweeping the judge's mixture, matching KDA-internal's
    fixed-shape variant.
    """

    official = [e for e in entries if e["suite"] == "official"]
    shapes = {(e["axes"]["num_heads"], e["axes"]["total_tokens"]) for e in official}
    if len(official) == 1 and len(shapes) == 1:
        return next(iter(shapes))
    return None


# Every comparison in this suite uses KDA-internal's semantics: the absolute
# tolerance is scaled by each output's reference RMS, and the returned tensors
# must be densely materialized with the reference's dtype, device and count --
# a candidate that omits ``final_state`` fails as "wrong number of outputs"
# rather than silently scoring on ``output`` alone.
_compare_outputs = functools.partial(compare_outputs, scale_atol_by_rms=True, strict_outputs=True)


def _passes_private_holdout(candidate, reference, device, shape=None):
    """Run the holdout with the configured salt or the local default."""

    from .holdout import verify_holdout

    return verify_holdout(
        candidate,
        reference,
        device,
        atol=ATOL,
        rtol=RTOL,
        required_matched_ratio=REQUIRED_MATCHED_RATIO,
        max_rel_l2=MAX_REL_L2,
        scale_atol_by_rms=True,
        repeated_keys_tolerance=CORRELATED_TOLERANCE,
        shape=shape,
    )


def run_suite(
    config: BenchConfig | None = None,
    candidate_fn=None,
    workloads=None,
    candidate_prepare_fn=None,
    holdout_shape=None,
):
    """Run the correctness gate and timing sweep for this task.

    Correctness is judged against the task definition's independent oracle
    (FLA's Triton ``chunk_kda``), while latency is measured against FlashKDA's
    fused CUTLASS backend in ``baseline.py``. Both reference implementations
    return the output and the final recurrent state, and both are scored.
    """

    from .baseline import run as baseline_forward

    cfg = config or default_config()
    device = choose_device(cfg.device)
    reference_forward = load_task_reference(TASK_NAME)
    entries = _resolve_entry_references(make_workloads(cfg) if workloads is None else workloads)
    # The task's registered shape mode determines holdout scope.
    if holdout_shape is None and cfg.shape_mode.lower() in ("max", "pinned"):
        holdout_shape = _pinned_shape(entries)
    candidate = candidate_fn
    if candidate is None:
        candidate = load_candidate_override(
            "BENCH_KDA_FORWARD_KERNEL",
            baseline_forward,
            "kda_forward_candidate",
        )
    if not _passes_private_holdout(
        _holdout_callable(candidate, candidate_prepare_fn),
        reference_forward,
        device,
        holdout_shape,
    ):
        # Report the refusal as a failing row rather than an empty result: a
        # consumer that sees no rows cannot tell a rejected candidate from a
        # suite that ran nothing.
        return [_holdout_failure_row()]
    return run_benchmark(
        name="kda_forward_k128_v128_bf16",
        workloads=entries,
        make_inputs=make_inputs,
        baseline_fn=baseline_forward,
        reference_fn=reference_forward,
        timing_baseline_fn=baseline_forward,
        compare_fn=_compare_outputs,
        candidate_fn=candidate,
        candidate_prepare_fn=candidate_prepare_fn,
        device=device,
        warmup=cfg.warmup,
        iters=cfg.iters,
        trials=cfg.trials,
        atol=ATOL,
        rtol=RTOL,
        required_matched_ratio=REQUIRED_MATCHED_RATIO,
        required_rms_error_ratios=REQUIRED_RMS_ERROR_RATIOS,
        group_axis="num_heads",
        correctness_runs=cfg.correctness_runs,
        check_after_timing=cfg.check_after_timing,
        require_repeatable_outputs=cfg.require_repeatable_outputs,
    )


def _resolve_entry_references(entries):
    """Bind each entry's named reference to this module's callable.

    Workload rows travel to the benchmark server as JSON, so a probe that needs
    an oracle other than the suite reference names it instead of carrying it.
    """

    resolved = []
    for entry in entries:
        name = entry.get("reference")
        if name is None:
            resolved.append(entry)
            continue
        if name != "recurrence_forward":
            raise ValueError(f"unknown entry reference {name!r}")
        resolved.append({**entry, "reference_fn": recurrence_forward})
    return resolved


def _holdout_failure_row():
    """One failing row shaped like run_benchmark's, with the probes kept opaque."""

    return {
        "suite": "holdout",
        "id": "holdout",
        "axes": {"total_tokens": 4096},
        "passed": False,
        "verdict": "FAIL",
        "note": "private holdout: numerical mismatch",
        "correctness_checks": 0,
        "max_abs": float("inf"),
        "max_rel": float("inf"),
        "max_rms_ratio": float("inf"),
        "matched": 0.0,
        "baseline_ms": None,
        "kernel_ms": None,
        "speedup": None,
        "timer": None,
        "timed": False,
    }


def _holdout_callable(candidate, candidate_prepare_fn):
    """Present the candidate to the holdout as a plain ``run(*args)`` callable.

    A TIRx candidate is a (prepare, run) pair so the harness can bind and
    compile outside the timed path; the holdout only ever calls it once per
    probe, so binding per call is what it needs.
    """

    if candidate_prepare_fn is None:
        return candidate

    def run(*args):
        return candidate(*candidate_prepare_fn(*args))

    return run


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    run_suite()


if __name__ == "__main__":
    main()
