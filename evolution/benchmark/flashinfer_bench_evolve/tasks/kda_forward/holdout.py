"""Salted correctness probes for KDA forward, ported from KDA-internal's judge.

The probe layout, deep-gate modes and key modes are the judge's; only the salt
differs. ``BENCH_INPUT_SALT`` selects it: the authoritative judge sets its own
secret value, and this checkout falls back to ``_LOCAL_SALT`` so the probes
actually run locally. A local run therefore exercises the same shapes, deep
modes and distributions as the judge, with different random values -- it is a
faithful rehearsal, not a way to predict the judge's exact inputs.
"""

from __future__ import annotations

import hashlib
import hmac
import math
import os
import random
from itertools import accumulate

import torch

from ...benchmark_common import clone_args, compare_outputs

_HEAD_DIM = 128
_DISTRIBUTIONS = ("normal", "uniform", "rademacher", "laplace")
# Which probes also drive part of the gate into the operator's saturated regime
# (see _apply_deep_mode); the assignment to probes is secret like everything else.
_DEEP_MODES = ("none", "none", "channels", "bands", "all", "random")
# A shape-pinned variant runs five probes: one of each deep mode.
_PINNED_DEEP_MODES = ("none", "channels", "bands", "all", "random")
# The judge keeps its salt secret; locally we still want the probes to run.
_LOCAL_SALT = "tirx-kda-forward-local-holdout-v1"


def _input_salt() -> str:
    return os.environ.get("BENCH_INPUT_SALT") or _LOCAL_SALT


def _secret_seed(label: str) -> int | None:
    salt = _input_salt()
    if not salt:
        return None
    digest = hmac.new(
        salt.encode(), f"kda-forward-holdout\0{label}".encode(), hashlib.sha256
    ).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


def _partition(total_tokens: int, num_seqs: int, rng: random.Random) -> tuple[int, ...]:
    if num_seqs == 1:
        return (total_tokens,)
    minimum = 64
    remaining = total_tokens - minimum * num_seqs
    weights = [rng.random() + 0.125 for _ in range(num_seqs)]
    extras = [int(remaining * weight / sum(weights)) for weight in weights]
    extras[-1] += remaining - sum(extras)
    return tuple(minimum + extra for extra in extras)


def _holdout_specs(shape: tuple[int, int] | None = None) -> tuple[dict, ...]:
    """Salted probe specs; ``shape`` pins every probe to one single-sequence
    (num_heads, total_tokens) so only the values and gate parameters vary."""
    layout_seed = _secret_seed("layout")
    if layout_seed is None:
        return ()
    rng = random.Random(layout_seed)
    heads = [64, 64, 96, 96, 64, 96]
    sequence_counts = [1, 3, 5, 7]
    distribution_offsets = list(range(len(_DISTRIBUTIONS)))
    rng.shuffle(heads)
    rng.shuffle(sequence_counts)
    # The full suite keeps two 8192-token packed probes at the end.
    sequence_counts.extend((6, 8))
    rng.shuffle(distribution_offsets)
    deep_modes = list(_PINNED_DEEP_MODES if shape is not None else _DEEP_MODES)
    rng.shuffle(deep_modes)
    # One of the weak-decay probes repeats a secret handful of keys instead of
    # drawing them (see _apply_key_mode); which one is secret too.
    repeated = deep_modes.index("none")

    specs = []
    probe_count = len(deep_modes)
    for index in range(probe_count):
        if shape is not None:
            total_tokens = shape[1]
        elif index >= 4:
            total_tokens = 8192
        else:
            total_tokens = 4097 + 512 * index + rng.randrange(512)
        locations = tuple(rng.uniform(-0.35, 0.35) for _ in range(6))
        spreads = tuple(rng.uniform(0.12, 0.75) for _ in range(6))
        dt_min = 10 ** rng.uniform(-4.0, -2.7)
        dt_max = 10 ** rng.uniform(-1.5, -0.6)
        specs.append(
            {
                "num_heads": heads[index] if shape is None else shape[0],
                "seq_lens": (
                    _partition(total_tokens, sequence_counts[index], rng)
                    if shape is None
                    else (shape[1],)
                ),
                "distribution_offset": distribution_offsets[index % len(distribution_offsets)],
                "locations": locations,
                "spreads": spreads,
                "a_min": rng.uniform(0.5, 3.0),
                "a_max": rng.uniform(10.0, 24.0),
                "dt_min": dt_min,
                "dt_max": max(dt_max, dt_min * 8.0),
                "value_seed": _secret_seed(f"values-{index}"),
                "deep_mode": deep_modes[index],
                "deep_fraction": rng.uniform(0.1, 0.6),
                "deep_band": rng.randrange(48, 200),
                "key_mode": "repeated" if index == repeated else "random",
                "key_count": rng.randrange(1, 4),
            }
        )
    return tuple(specs)


def _deep_values(shape, low, high, dtype, device, generator):
    values = torch.rand(shape, dtype=torch.float32, device=device, generator=generator)
    return values.mul_(high - low).add_(low).to(dtype)


def _apply_deep_mode(spec: dict, g, dt_bias, A_log, generator) -> None:
    """Drive part of the gate into the operator's saturated regime, in place.

    Deep positions get ``g`` in [2, 3] and ``dt_bias`` in [-0.5, 0], so with any
    ``exp(A_log) >= 0.5`` the channel decays by more than 4.9 bits per token:
    far beyond a factorisation that is only exact for realistic gates, and never
    the same pattern as the public stress probes. ``channels`` saturates a
    secret subset of channels for every token, ``bands`` alternates saturated
    and realistic token bands, ``all`` saturates everything, ``random`` draws
    ``A_log``, ``dt_bias`` and ``g`` standard normal (the recipe of FLA's own
    tests), mixing every depth across the channels of one head.
    """
    mode = spec.get("deep_mode", "none")
    if mode == "none":
        return
    device = g.device
    heads = dt_bias.shape[0] // _HEAD_DIM
    if mode == "random":
        A_log.normal_(generator=generator)
        dt_bias.normal_(generator=generator)
        g.copy_(torch.randn(g.shape, dtype=torch.float32, device=device, generator=generator))
        return
    gate = _deep_values(g.shape, 2.0, 3.0, g.dtype, device, generator)
    bias = _deep_values(dt_bias.shape, -0.5, 0.0, dt_bias.dtype, device, generator)
    if mode == "all":
        g.copy_(gate)
        dt_bias.copy_(bias)
    elif mode == "channels":
        deep = (
            torch.rand((heads, _HEAD_DIM), device=device, generator=generator)
            < spec["deep_fraction"]
        )
        g.copy_(torch.where(deep[None, None], gate, g))
        dt_bias.copy_(torch.where(deep.reshape(-1), bias, dt_bias))
    elif mode == "bands":
        tokens = torch.arange(g.shape[1], device=device)
        deep = (tokens // spec["deep_band"]) % 2 == 0
        g.copy_(torch.where(deep[None, :, None, None], gate, g))
        dt_bias.copy_(bias)
    else:
        raise ValueError(f"unknown deep mode {mode!r}")


def _apply_key_mode(spec: dict, q, k, g, beta, generator) -> None:
    """Repeat a secret handful of keys with no decay and a beta near one, in place.

    ``repeated`` draws ``key_count`` directions per head and gives every token
    one of them as both its key and its query, with ``beta`` logits in [2, 4]
    and ``g`` in [-3, -1.5], so the channels barely decay and the intra-chunk
    system is dense in identical rows: the regime in which an inverse evaluated
    as a low-precision series loses its cancellation. ``random`` leaves the
    drawn values alone.
    """
    mode = spec.get("key_mode", "random")
    if mode == "random":
        return
    if mode != "repeated":
        raise ValueError(f"unknown key mode {mode!r}")
    device = k.device
    _, total_tokens, heads, head_dim = k.shape
    directions = torch.randn(
        (spec["key_count"], heads, head_dim),
        dtype=torch.float32,
        device=device,
        generator=generator,
    )
    choice = torch.randint(
        0, spec["key_count"], (total_tokens,), device=device, generator=generator
    )
    keys = directions[choice].to(k.dtype)[None]
    k.copy_(keys)
    q.copy_(keys)
    beta.copy_(_deep_values(beta.shape, 2.0, 4.0, beta.dtype, device, generator))
    g.copy_(_deep_values(g.shape, -3.0, -1.5, g.dtype, device, generator))


def _draw(spec: dict, slot: int, shape, dtype, device, generator):
    location = spec["locations"][slot]
    spread = spec["spreads"][slot]
    distribution = _DISTRIBUTIONS[(spec["distribution_offset"] + slot) % len(_DISTRIBUTIONS)]
    if distribution == "normal":
        values = (
            torch.randn(shape, dtype=torch.float32, device=device, generator=generator)
            .mul_(spread)
            .add_(location)
        )
    elif distribution == "uniform":
        values = (
            torch.rand(shape, dtype=torch.float32, device=device, generator=generator)
            .mul_(2.0 * spread)
            .add_(location - spread)
        )
    elif distribution == "rademacher":
        values = (
            torch.randint(
                0,
                2,
                shape,
                dtype=torch.int8,
                device=device,
                generator=generator,
            )
            .float()
            .mul_(2.0)
            .sub_(1.0)
            .mul_(spread)
            .add_(location)
        )
    else:
        centered = (
            torch.rand(shape, dtype=torch.float32, device=device, generator=generator) - 0.5
        ).clamp_(-0.499, 0.499)
        values = location - spread * torch.sign(centered) * torch.log1p(-2.0 * torch.abs(centered))
    return values.clamp_(-3.0, 3.0).to(dtype)


def _make_inputs(spec: dict, device):
    num_heads = spec["num_heads"]
    seq_lens = spec["seq_lens"]
    total_tokens = sum(seq_lens)
    generator = torch.Generator(device=device)
    generator.manual_seed(spec["value_seed"])
    shape = (1, total_tokens, num_heads, _HEAD_DIM)
    q = _draw(spec, 0, shape, torch.bfloat16, device, generator)
    k = _draw(spec, 1, shape, torch.bfloat16, device, generator)
    v = _draw(spec, 2, shape, torch.bfloat16, device, generator)
    g = _draw(spec, 3, shape, torch.bfloat16, device, generator)
    beta = _draw(
        spec,
        4,
        (1, total_tokens, num_heads),
        torch.bfloat16,
        device,
        generator,
    )
    A_log = torch.log(
        torch.rand(num_heads, dtype=torch.float32, device=device, generator=generator).mul_(
            spec["a_max"] - spec["a_min"]
        )
        + spec["a_min"]
    )
    dt = torch.exp(
        torch.rand(
            num_heads * _HEAD_DIM,
            dtype=torch.float32,
            device=device,
            generator=generator,
        ).mul_(math.log(spec["dt_max"]) - math.log(spec["dt_min"]))
        + math.log(spec["dt_min"])
    ).clamp_(min=1e-5)
    dt_bias = dt + torch.log(-torch.expm1(-dt))
    _apply_deep_mode(spec, g, dt_bias, A_log, generator)
    _apply_key_mode(spec, q, k, g, beta, generator)
    cu_seqlens = None
    if len(seq_lens) > 1:
        cu_seqlens = torch.tensor([0, *accumulate(seq_lens)], device=device, dtype=torch.long)
    initial_state = _draw(
        spec,
        5,
        (len(seq_lens), num_heads, _HEAD_DIM, _HEAD_DIM),
        torch.float32,
        device,
        generator,
    )
    return (
        q,
        k,
        v,
        g,
        beta,
        A_log,
        dt_bias,
        1.0 / math.sqrt(_HEAD_DIM),
        initial_state,
        cu_seqlens,
    )


def verify_holdout(
    candidate_fn,
    reference_fn,
    device,
    *,
    atol: float,
    rtol: float,
    required_matched_ratio: float,
    max_rel_l2: float,
    scale_atol_by_rms: bool = False,
    repeated_keys_tolerance: dict | None = None,
    shape: tuple[int, int] | None = None,
) -> bool:
    """Return whether the candidate matched the reference on every probe.

    ``repeated_keys_tolerance`` overrides ``atol``, ``rtol`` and ``max_rel_l2``
    for the probe that repeats keys (ill-conditioned by construction, like the
    public ``correlated`` probe); the other probes use the suite's tolerances."""
    specs = _holdout_specs(shape)
    if not specs:
        return True

    default_ratios = (max_rel_l2, max_rel_l2)
    passed = True
    for spec in specs:
        args = _make_inputs(spec, device)
        with torch.no_grad():
            reference = reference_fn(*clone_args(args))
        torch.cuda.synchronize(device)
        try:
            with torch.no_grad():
                candidate = candidate_fn(*clone_args(args))
            torch.cuda.synchronize(device)
        except Exception:
            passed = False
            break
        tolerance = {
            "atol": atol,
            "rtol": rtol,
            "required_rms_error_ratios": default_ratios,
        }
        if spec["key_mode"] == "repeated" and repeated_keys_tolerance is not None:
            tolerance.update(repeated_keys_tolerance)
        if not compare_outputs(
            candidate,
            reference,
            tolerance["atol"],
            tolerance["rtol"],
            required_matched_ratio,
            tolerance["required_rms_error_ratios"],
            scale_atol_by_rms=scale_atol_by_rms,
            strict_outputs=True,
        )[0]:
            passed = False
            break

    print(f"holdout correctness: {'PASS' if passed else 'FAIL numerical mismatch'}")
    return passed
