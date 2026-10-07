"""CUDA-free NumSim cases for recurrent attention kernels."""

from __future__ import annotations

import math
from typing import Any
from unittest.mock import patch

import numpy as np

from tests.numsim.support._tirx_kernels import load_tirx_kernel
from tirx_harness.numsim.cases import (
    ComparisonRegion,
    ComparisonSpec,
    NumSimCase,
    TensorMap,
)

native_kda_forward = load_tirx_kernel("kda_forward_portfolio_multishape")
recurrent_kda_decode_grouped = load_tirx_kernel("recurrent_kda_decode_grouped")
recurrent_kda_decode_one_warp = load_tirx_kernel("recurrent_kda_decode_one_warp")
gdn_decode_bf16_ilp4 = load_tirx_kernel("gdn_decode_bf16_ilp4")
gdn_decode_bf16_wide_vec_mtp = load_tirx_kernel("gdn_decode_bf16_wide_vec_mtp")
gdn_decode_bf16_wide_vec_t1 = load_tirx_kernel("gdn_decode_bf16_wide_vec_t1")
gdn_decode_fp32_mtp_warp = load_tirx_kernel("gdn_decode_fp32_mtp_warp")
gdn_cp_prefill_sm100 = load_tirx_kernel("gdn_cp_prefill_sm100")
gdn_prefill_sm100 = load_tirx_kernel("gdn_prefill_sm100")

_HEAD_DIM = 128
_FLASHKDA_LOWER_BOUND = -5.0
_GDN_DESCRIPTOR_BYTES_PER_CTA = 512
_L2_EPS = np.float32(1.0e-6)


def _float32_to_bfloat16_bits(values: Any) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    bits = values.view(np.uint32)
    rounding_bias = np.uint32(0x7FFF) + ((bits >> np.uint32(16)) & np.uint32(1))
    return ((bits + rounding_bias) >> np.uint32(16)).astype(np.uint16)


def _bfloat16_bits_to_float32(values: Any) -> np.ndarray:
    bits = np.asarray(values, dtype=np.uint16).astype(np.uint32) << np.uint32(16)
    return bits.view(np.float32)


def _round_to_bfloat16(values: Any) -> np.ndarray:
    return _bfloat16_bits_to_float32(_float32_to_bfloat16_bits(values))


def _sigmoid(values: Any) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    return np.float32(1.0) / (np.float32(1.0) + np.exp(-values))


def kda_decode_numpy_reference(
    q_bits: np.ndarray,
    k_bits: np.ndarray,
    v_bits: np.ndarray,
    g_bits: np.ndarray,
    beta_bits: np.ndarray,
    *,
    seq_lens: tuple[int, ...],
    scale: float,
    precomputed_gate: bool,
    a_log: np.ndarray | None = None,
    dt_bias: np.ndarray | None = None,
    lower_bound: float | None = None,
) -> np.ndarray:
    """Dense FP32 oracle shared by the latest canonical KDA decoders."""

    q = _bfloat16_bits_to_float32(q_bits)
    k = _bfloat16_bits_to_float32(k_bits)
    v = _bfloat16_bits_to_float32(v_bits)
    g = _bfloat16_bits_to_float32(g_bits)
    beta = _bfloat16_bits_to_float32(beta_bits)
    num_value_heads = v.shape[1]
    num_heads = q.shape[1]
    if num_value_heads % num_heads:
        raise ValueError("KDA oracle requires value heads to be divisible by query heads")
    head_ratio = num_value_heads // num_heads

    if not precomputed_gate:
        if a_log is None or dt_bias is None:
            raise ValueError("in-kernel KDA gate requires A_log and dt_bias")
        a_log = np.asarray(a_log, dtype=np.float32)
        dt_bias = np.asarray(dt_bias, dtype=np.float32).reshape(num_heads, _HEAD_DIM)

    state = np.zeros((len(seq_lens), num_value_heads, _HEAD_DIM, _HEAD_DIM), dtype=np.float32)
    output = np.zeros_like(v, dtype=np.float32)
    token_base = 0
    for sequence, length in enumerate(seq_lens):
        for token in range(token_base, token_base + length):
            for value_head in range(num_value_heads):
                head = value_head // head_ratio
                q_row = q[token, head]
                k_row = k[token, head]
                q_norm = q_row * (
                    np.float32(scale) / np.sqrt(np.sum(q_row * q_row, dtype=np.float32) + _L2_EPS)
                )
                k_norm = k_row / np.sqrt(np.sum(k_row * k_row, dtype=np.float32) + _L2_EPS)
                if precomputed_gate:
                    decay = np.exp(g[token, value_head])
                else:
                    gate_input = g[token, value_head] + dt_bias[head]
                    gate_scale = np.exp(a_log[head])
                    if lower_bound is None:
                        log_gate = -gate_scale * np.logaddexp(np.float32(0.0), gate_input)
                    else:
                        log_gate = np.float32(lower_bound) / (
                            np.float32(1.0) + np.exp(-gate_scale * gate_input)
                        )
                    decay = np.exp(log_gate)

                current = state[sequence, value_head] * decay[None, :]
                predicted = current @ k_norm
                residual = beta[token, value_head] * (v[token, value_head] - predicted)
                current += residual[:, None] * k_norm[None, :]
                state[sequence, value_head] = current
                output[token, value_head] = current @ q_norm
        token_base += length
    return output


def gdn_decode_bf16_numpy_reference(
    q_bits: np.ndarray,
    k_bits: np.ndarray,
    v_bits: np.ndarray,
    a_bits: np.ndarray,
    b_bits: np.ndarray,
    a_log: np.ndarray,
    dt_bias: np.ndarray,
    *,
    scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Token-serial GDN oracle for BF16 inputs and BF16/FP32 state ports."""

    q = _bfloat16_bits_to_float32(q_bits)
    k = _bfloat16_bits_to_float32(k_bits)
    v = _bfloat16_bits_to_float32(v_bits)
    a = _bfloat16_bits_to_float32(a_bits)
    b = _bfloat16_bits_to_float32(b_bits)
    a_log = np.asarray(a_log, dtype=np.float32)
    dt_bias = np.asarray(dt_bias, dtype=np.float32)
    batch, seq_len, num_heads, _ = q.shape
    num_value_heads = v.shape[2]
    if num_value_heads % num_heads:
        raise ValueError("GDN oracle requires value heads to be divisible by query heads")
    head_ratio = num_value_heads // num_heads
    state = np.zeros((batch, num_value_heads, _HEAD_DIM, _HEAD_DIM), dtype=np.float32)
    output = np.zeros_like(v, dtype=np.float32)
    for batch_index in range(batch):
        for token in range(seq_len):
            for value_head in range(num_value_heads):
                head = value_head // head_ratio
                q_row = q[batch_index, token, head]
                k_row = k[batch_index, token, head]
                q_norm = q_row * (
                    np.float32(scale) / np.sqrt(np.sum(q_row * q_row, dtype=np.float32) + _L2_EPS)
                )
                k_norm = k_row / np.sqrt(np.sum(k_row * k_row, dtype=np.float32) + _L2_EPS)
                gate_input = a[batch_index, token, value_head] + dt_bias[value_head]
                softplus = np.logaddexp(np.float32(0.0), gate_input)
                decay = np.exp(-np.exp(a_log[value_head]) * softplus)
                beta = np.float32(1.0) / (
                    np.float32(1.0) + np.exp(-b[batch_index, token, value_head])
                )
                current = state[batch_index, value_head] * decay
                predicted = current @ k_norm
                residual = beta * (v[batch_index, token, value_head] - predicted)
                current += residual[:, None] * k_norm[None, :]
                state[batch_index, value_head] = current
                output[batch_index, token, value_head] = current @ q_norm
    return output, state


def _tensor_map(
    base: np.ndarray,
    *,
    dtype: str,
    global_shape: tuple[int, ...],
    global_strides: tuple[int, ...],
    box_shape: tuple[int, ...],
    swizzle: str | None,
) -> np.ndarray:
    return TensorMap(
        base=base,
        dtype=dtype,
        global_shape=global_shape,
        global_strides=global_strides,
        box_shape=box_shape,
        element_strides=(1,) * len(global_shape),
        swizzle=swizzle,
        interleave=None,
        fill_mode="none",
    ).numpy()


def _agent_kda_tensor_map(
    base: np.ndarray, seq_len: int, num_heads: int, *, beta: bool = False, rows: int = 64
) -> np.ndarray:
    import torch

    # Consume the production encoder's layout at its CUDA boundary so the
    # fixture cannot silently invent a different rank, stride, or tile box.
    descriptors = []

    def encode(_destination, dtype, rank, address, *layout):
        assert address.value == base.ctypes.data
        assert len(layout) == 4 * rank + 3
        interleave, swizzle, _promotion, fill = layout[-4:]
        assert interleave == 0 and fill == 0
        descriptors.append(
            TensorMap(
                base=base,
                dtype=dtype,
                global_shape=tuple(layout[:rank]),
                global_strides=tuple(layout[rank : 2 * rank - 1]),
                box_shape=tuple(layout[2 * rank - 1 : 3 * rank - 1]),
                element_strides=tuple(layout[3 * rank - 1 : 4 * rank - 1]),
                swizzle={0: None, 1: "32B", 2: "64B", 3: "128B"}[swizzle],
            ).numpy()
        )

    with patch.object(native_kda_forward, "_TMAP_ENCODE", encode):
        encoder = (
            native_kda_forward._encode_beta_map
            if beta
            else native_kda_forward._encode_map
        )
        if beta:
            encoder(torch.from_numpy(base), seq_len, num_heads)
        else:
            encoder(torch.from_numpy(base), seq_len, num_heads, rows=rows)
    [descriptor] = descriptors
    return descriptor


def prepare_native_kda_forward_case() -> NumSimCase:
    """Exercise both packed sequences of one head through the native fused route."""

    seq_lens = (64, 64)
    seq_len = sum(seq_lens)
    num_heads = 64
    active_heads = 1
    values_shape = (seq_len, num_heads, _HEAD_DIM)
    one = _float32_to_bfloat16_bits(np.array([1.0], dtype=np.float32))[0]
    # The read-only q/k/v/g ports share one valid backing. The e0 input
    # keeps the recurrence analytic while exercising nonzero data.
    qkvg = np.zeros(values_shape, dtype=np.uint16)
    qkvg[:, :active_heads, 0] = one
    beta = np.zeros((seq_len, num_heads), dtype=np.uint16)
    a_log = np.zeros((num_heads,), dtype=np.float32)
    # Match the production generator's inverse-softplus parameterization.
    # A 0.01 base step keeps every 64-token chunk's cumulative decay finite
    # while the shared nonzero g backing still exercises gate evaluation.
    dt = np.float32(0.01)
    dt_bias_value = np.float32(dt + math.log(-math.expm1(-float(dt))))
    dt_bias = np.full((num_heads, _HEAD_DIM), dt_bias_value, dtype=np.float32)
    initial_state = np.zeros((len(seq_lens), num_heads, _HEAD_DIM, _HEAD_DIM), dtype=np.float32)
    final_state = np.full_like(initial_state, np.nan)
    output = np.full(values_shape, 0x7FC0, dtype=np.uint16)

    qkvg_binding = qkvg.reshape(-1)
    output_binding = output.reshape(-1)

    normalized = np.float32(1.0 / math.sqrt(1.0 + 1.0e-6))
    gate = np.float32(-5.0 / (1.0 + math.exp(-(1.0 + float(dt_bias_value)))))
    decay = np.float32(math.exp(gate))
    expected_scalar = np.empty(seq_len, dtype=np.float32)
    expected_final_state = np.zeros_like(initial_state)
    token_base = 0
    for sequence, length in enumerate(seq_lens):
        state = np.float32(0.0)
        for token in range(token_base, token_base + length):
            current = decay * state
            residual = np.float32(0.5) * (np.float32(1.0) - current * normalized)
            state = current + residual * normalized
            expected_scalar[token] = state * normalized * np.float32(1.0 / math.sqrt(_HEAD_DIM))
        expected_final_state[sequence, 0, 0, 0] = state
        token_base += length
    # The production descriptor splits each head into two 64-element halves;
    # snapshots reverse its innermost-first (64, T, 2H) dimensions.
    expected_output = np.zeros((2 * active_heads, seq_len, 64), dtype=np.float32)
    expected_output[0, :, 0] = expected_scalar
    output_map = _agent_kda_tensor_map(output_binding, seq_len, num_heads, rows=32)
    beta_map = _agent_kda_tensor_map(beta.reshape(-1), seq_len, num_heads, beta=True)
    args = {
        "q_map": _agent_kda_tensor_map(qkvg_binding, seq_len, num_heads),
        "k_map": _agent_kda_tensor_map(qkvg_binding, seq_len, num_heads),
        "v_map": _agent_kda_tensor_map(qkvg_binding, seq_len, num_heads),
        "g_map": _agent_kda_tensor_map(qkvg_binding, seq_len, num_heads),
        "beta_map": beta_map,
        "o_map": output_map,
        "out": output_binding,
        "A_log": a_log,
        "dt_bias": dt_bias.reshape(-1),
        "h0": initial_state.reshape(-1),
        "final_state": final_state.reshape(-1),
        "hand": np.zeros(num_heads * _HEAD_DIM * _HEAD_DIM, dtype=np.float32),
        "flags": np.zeros(num_heads + 1, dtype=np.int32),
        "cu": np.array([0, seq_lens[0], seq_len], dtype=np.int64),
        "nseq": np.int32(len(seq_lens)),
        "scale": np.float32(1.0 / math.sqrt(_HEAD_DIM)),
    }
    return NumSimCase(
        # The public packed dispatch returns one PrimFunc. Run its full 64-CTA
        # schedule so each CTA owns one complete head across both sequences.
        kernel=_specialize_runtime_scalars(
            native_kda_forward.get_kernel(num_heads=num_heads, seq_lens=seq_lens),
            {"num_ctas": num_heads},
        ),
        args=args,
        outputs={"output": "o_map", "final_state": "final_state"},
        reference=lambda: {
            "output": expected_output.copy(),
            "final_state": expected_final_state.reshape(-1).copy(),
        },
        comparisons={
            "output": ComparisonSpec(
                rtol=5e-2,
                atol=5e-4,
                actual_encoding="bfloat16",
                regions=(
                    ComparisonRegion(
                        actual=(slice(0, 2 * active_heads), slice(None), slice(None)),
                        expected=(slice(None), slice(None), slice(None)),
                    ),
                ),
            ),
            "final_state": ComparisonSpec(rtol=5e-2, atol=5e-4),
        },
    )


def prepare_native_kda_fixed_case() -> NumSimCase:
    """Exercise both kernels in the native fixed-sequence dispatch."""

    import torch

    module = native_kda_forward
    length, heads, dim, chunk = 64, 64, _HEAD_DIM, module.C
    items = length // chunk * heads
    one = _float32_to_bfloat16_bits(np.array([1.0], dtype=np.float32))[0]
    qkvg = np.zeros((length, heads, dim), dtype=np.uint16)
    qkvg[:, 0, 0] = one
    qkvg = qkvg.reshape(-1)
    beta = np.zeros(length * heads, dtype=np.uint16)
    a_log = np.zeros(heads, dtype=np.float32)
    dt = np.float32(0.01)
    dt_bias_value = np.float32(dt + math.log(-math.expm1(-float(dt))))
    dt_bias = np.full(heads * dim, dt_bias_value, dtype=np.float32)
    initial_state = np.zeros(heads * dim * dim, dtype=np.float32)
    final_state = np.full_like(initial_state, np.nan)
    output = np.full(length * heads * dim, 0x7FC0, dtype=np.uint16)
    vec = np.zeros(items * module.VEC_F32, dtype=np.float32)
    kbar = np.zeros(items * chunk * dim, dtype=np.uint16)
    qt = np.zeros_like(kbar)
    t1 = np.zeros(items * chunk * chunk, dtype=np.uint16)
    aqk = np.zeros_like(t1)
    w1 = np.zeros(items * dim * chunk, dtype=np.uint16)
    flags = np.zeros(items, dtype=np.int32)

    def descriptor(backing, encoder, *encoder_args):
        def encode(tensor, dims, strides_bytes, box, swizzle_code):
            return TensorMap(
                base=tensor.numpy(), dtype="bfloat16",
                global_shape=tuple(dims), global_strides=tuple(strides_bytes),
                box_shape=tuple(box), element_strides=(1, 1, 1),
                swizzle={2: "64B", 3: "128B"}[swizzle_code],
            ).numpy()

        with patch.object(module, "_encode", encode):
            return encoder(torch.from_numpy(backing), *encoder_args)

    input_map = descriptor(qkvg, module._encode_thd, length, heads)
    fixed = module.get_kernel(num_heads=heads, seq_lens=(length,))
    front = _specialize_runtime_scalars(fixed["kda_front"], {"num_ctas": heads})
    front_args = {
        "q": qkvg, "k": qkvg, "g": qkvg, "beta": beta,
        "a_log": a_log, "dt_bias": dt_bias, "vec": vec,
        "kbar_g": kbar, "qt_g": qt, "t1_g": t1, "aqk_g": aqk,
        "w1_g": w1, "flags": flags,
        "q_map": input_map, "k_map": input_map, "g_map": input_map,
        "item_base": np.int32(0), "num_items": np.int32(items),
        "items_per_cta": np.int32(items // heads), "do_signal": np.int32(0),
    }
    chain_args = {
        "v": qkvg, "state_in": initial_state, "state_out": final_state,
        "out": output, "vec": vec, "kbar_g": kbar, "qt_g": qt,
        "t1_g": t1, "aqk_g": aqk, "w1_g": w1,
        "v_map": input_map,
        "kbar_map": descriptor(kbar, module._encode_tile, chunk, dim, items),
        "qt_map": descriptor(qt, module._encode_tile, chunk, dim, items),
        "t1_map": descriptor(t1, module._encode_tile, chunk, chunk, items),
        "aqk_map": descriptor(aqk, module._encode_tile, chunk, chunk, items),
        "w1_map": descriptor(w1, module._encode_tile, dim, chunk, items),
        "o_map": descriptor(output, module._encode_thd, length, heads),
        "flags": flags, "scale": np.float32(1.0 / math.sqrt(dim)),
        "num_chunks": np.int32(length // chunk),
        "flag_from": np.int32(items), "flag_target": np.int32(1),
    }
    args = {f"k0:{name}": value for name, value in front_args.items()}
    args.update({f"k1:{name}": value for name, value in chain_args.items()})

    normalized = np.float32(1.0 / math.sqrt(1.0 + 1.0e-6))
    gate = np.float32(-5.0 / (1.0 + math.exp(-(1.0 + float(dt_bias_value)))))
    decay = np.float32(math.exp(gate))
    expected_output = np.zeros((length, heads, dim), dtype=np.float32)
    expected_final_state = np.zeros((heads, dim, dim), dtype=np.float32)
    state = np.float32(0.0)
    for token in range(length):
        current = decay * state
        residual = np.float32(0.5) * (np.float32(1.0) - current * normalized)
        state = current + residual * normalized
        expected_output[token, 0, 0] = state * normalized * np.float32(1.0 / math.sqrt(dim))
    expected_final_state[0, 0, 0] = state
    return NumSimCase(
        kernel=(front, fixed["kda_chain"]), args=args,
        outputs=("k1:out", "k1:state_out"),
        reference=lambda: {
            "k1:out": expected_output.reshape(-1).copy(),
            "k1:state_out": expected_final_state.reshape(-1).copy(),
        },
        comparisons={
            "k1:out": ComparisonSpec(rtol=5e-2, atol=5e-4, actual_encoding="bfloat16"),
            "k1:state_out": ComparisonSpec(rtol=5e-2, atol=5e-4),
        },
    )


def gdn_numpy_reference(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    gate: np.ndarray,
    beta: np.ndarray,
    initial_state: np.ndarray,
    *,
    seq_lens: tuple[int, ...],
    scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Token-serial gated delta-rule oracle for the HV-multiple-of-HQ path."""

    q = np.asarray(q, dtype=np.float16).astype(np.float32)
    k = np.asarray(k, dtype=np.float16).astype(np.float32)
    v = np.asarray(v, dtype=np.float16).astype(np.float32)
    gate = np.asarray(gate, dtype=np.float32)
    beta = np.asarray(beta, dtype=np.float32)
    state = np.asarray(initial_state, dtype=np.float32).copy()
    num_q_heads = q.shape[1]
    num_value_heads = v.shape[1]
    if num_value_heads % num_q_heads:
        raise ValueError("GDN oracle requires HV to be divisible by HQ")
    value_heads_per_q_head = num_value_heads // num_q_heads
    output = np.zeros_like(v, dtype=np.float32)
    start = 0
    for sequence, length in enumerate(seq_lens):
        for token in range(start, start + length):
            for value_head in range(num_value_heads):
                qk_head = value_head // value_heads_per_q_head
                current = gate[token, value_head] * state[sequence, value_head]
                predicted = current @ k[token, qk_head]
                residual = beta[token, value_head] * (v[token, value_head] - predicted)
                current += residual[:, None] * k[token, qk_head][None, :]
                state[sequence, value_head] = current
                output[token, value_head] = np.float32(scale) * (current @ q[token, qk_head])
        start += length
    return output.astype(np.float16), state


def _specialize_runtime_scalars(kernel: Any, values: dict[str, int]) -> Any:
    substitutions = {
        parameter: values[parameter.name] for parameter in kernel.params if parameter.name in values
    }
    return kernel.specialize(substitutions)


def _prepare_kda_decode_case(
    module: Any,
    config: dict[str, Any],
    *,
    num_sequences: int,
    num_tokens: int,
    num_heads: int,
    num_value_heads: int,
    precomputed_gate: bool,
    checkpoint_base: int,
) -> NumSimCase:
    kernel = module.get_kernel(**config)
    total_tokens = num_sequences * num_tokens
    q_values = np.zeros((total_tokens, num_heads, _HEAD_DIM), dtype=np.float32)
    k_values = np.zeros_like(q_values)
    q_values[..., 0] = np.float32(1.0)
    k_values[..., 0] = np.float32(1.0)
    v_values = np.zeros((total_tokens, num_value_heads, _HEAD_DIM), dtype=np.float32)
    base_vector = np.array([1, 2, 3, 4], dtype=np.float32)
    for token in range(total_tokens):
        for head in range(num_value_heads):
            v_values[token, head, :4] = base_vector * np.float32((token + 1) * (head + 1) / 64)
    g_values = np.zeros_like(v_values)
    beta_values = np.full((total_tokens, num_value_heads), np.float32(0.5), dtype=np.float32)
    q = _float32_to_bfloat16_bits(q_values)
    k = _float32_to_bfloat16_bits(k_values)
    v = _float32_to_bfloat16_bits(v_values)
    g = _float32_to_bfloat16_bits(g_values)
    beta = _float32_to_bfloat16_bits(beta_values)
    cu_seqlens = np.arange(0, total_tokens + 1, num_tokens, dtype=np.int32)
    state_indices = np.arange(total_tokens, dtype=np.int32) + np.int32(checkpoint_base)
    configured_state_slots = int(
        config.get("pool_size", num_sequences + int(config.get("pool_slack", 0)))
    )
    state_slots = max(configured_state_slots, int(np.max(state_indices)) + 1)
    state = np.zeros(state_slots * num_value_heads * _HEAD_DIM * _HEAD_DIM, dtype=np.uint16)
    output = np.zeros(total_tokens * num_value_heads * _HEAD_DIM, dtype=np.uint16)
    accepted = np.ones(num_sequences, dtype=np.int32)
    a_log = np.zeros(num_heads, dtype=np.float32)
    dt_bias = np.zeros((num_heads, _HEAD_DIM), dtype=np.float32)
    scale = float(1.0 / math.sqrt(_HEAD_DIM))
    lower_bound = config.get("lower_bound")
    expected = kda_decode_numpy_reference(
        q.reshape(total_tokens, num_heads, _HEAD_DIM),
        k.reshape(total_tokens, num_heads, _HEAD_DIM),
        v.reshape(total_tokens, num_value_heads, _HEAD_DIM),
        g.reshape(total_tokens, num_value_heads, _HEAD_DIM),
        beta.reshape(total_tokens, num_value_heads),
        seq_lens=(num_tokens,) * num_sequences,
        scale=scale,
        precomputed_gate=precomputed_gate,
        a_log=a_log,
        dt_bias=dt_bias,
        lower_bound=lower_bound,
    )

    q_binding = q.reshape(-1)
    k_binding = k.reshape(-1)
    v_binding = v.reshape(-1)
    g_binding = g.reshape(-1)
    beta_binding = beta.reshape(-1)
    state_binding = state
    output_binding = output
    args: dict[str, Any] = {
        "q": q_binding,
        "k": k_binding,
        "v": v_binding,
        "g": g_binding,
        "beta": beta_binding,
        "state": state_binding,
        "out": output_binding,
        "scale": scale,
    }
    parameter_names = {parameter.name for parameter in kernel.params}
    if "a_log" in parameter_names:
        args["a_log"] = a_log
    if "dt_bias" in parameter_names:
        args["dt_bias"] = dt_bias.reshape(-1)
    if "cu" in parameter_names:
        args["cu"] = cu_seqlens
    if "cu_seqlens" in parameter_names:
        args["cu_seqlens"] = cu_seqlens
    if "ssm_idx" in parameter_names:
        args["ssm_idx"] = state_indices
    if "ssm_state_indices" in parameter_names:
        args["ssm_state_indices"] = state_indices
    if "nat" in parameter_names:
        args["nat"] = accepted
    if "q_total" in parameter_names:
        args["q_total"] = total_tokens
    if "g_stride_q" in parameter_names:
        args["g_stride_q"] = num_value_heads * _HEAD_DIM
    if "state_slot_stride" in parameter_names:
        args["state_slot_stride"] = num_value_heads * _HEAD_DIM * _HEAD_DIM
    if "eps" in parameter_names:
        args["eps"] = float(_L2_EPS)
    if "lower_bound" in parameter_names:
        args["lower_bound"] = float(lower_bound)

    return NumSimCase(
        kernel=kernel,
        args=args,
        outputs=("out",),
        reference=lambda: {"out": expected.reshape(-1).copy()},
        comparisons={"out": ComparisonSpec(rtol=2e-2, atol=2e-3, actual_encoding="bfloat16")},
    )


def prepare_recurrent_kda_grouped_case() -> NumSimCase:
    return _prepare_kda_decode_case(
        recurrent_kda_decode_grouped,
        {
            "mode": "verify",
            "num_seqs": 1,
            "num_tokens": 2,
            "num_heads": 1,
            "num_value_heads": 1,
            "pool_size": 2,
            "lower_bound": _FLASHKDA_LOWER_BOUND,
            "empty_rows": 0,
            "orphan_tokens": 0,
        },
        num_sequences=1,
        num_tokens=2,
        num_heads=1,
        num_value_heads=1,
        precomputed_gate=False,
        checkpoint_base=0,
    )


def prepare_recurrent_kda_one_warp_case() -> NumSimCase:
    return _prepare_kda_decode_case(
        recurrent_kda_decode_one_warp,
        {
            "num_seqs": 8,
            "num_heads": 16,
            "num_value_heads": 16,
            "pool_size": 8,
            "lower_bound": _FLASHKDA_LOWER_BOUND,
        },
        num_sequences=8,
        num_tokens=1,
        num_heads=16,
        num_value_heads=16,
        precomputed_gate=False,
        checkpoint_base=0,
    )


def _prepare_gdn_decode_case(
    module: Any,
    config: dict[str, Any],
    *,
    seq_len: int,
    state_dtype: str = "bfloat16",
) -> NumSimCase:
    state_numpy_dtype = {
        "bfloat16": np.dtype(np.uint16),
        "float32": np.dtype(np.float32),
    }.get(state_dtype)
    if state_numpy_dtype is None:
        raise ValueError(f"unsupported GDN state dtype: {state_dtype}")
    batch = int(config["batch"])
    num_heads = int(config["num_heads"])
    num_value_heads = int(config["num_v_heads"])
    kernel = _specialize_runtime_scalars(module.get_kernel(**config), {"batch": batch})
    q_values = np.zeros((batch, seq_len, num_heads, _HEAD_DIM), dtype=np.float32)
    k_values = np.zeros_like(q_values)
    q_values[..., 0] = np.float32(1.0)
    k_values[..., 0] = np.float32(1.0)
    v_values = np.zeros((batch, seq_len, num_value_heads, _HEAD_DIM), dtype=np.float32)
    base_vector = np.array([1, 2, 3, 4], dtype=np.float32)
    for batch_index in range(batch):
        for token in range(seq_len):
            for head in range(num_value_heads):
                v_values[batch_index, token, head, :4] = base_vector * np.float32(
                    (batch_index + 1) * (token + 1) * (head + 1) / 64
                )
    a_values = np.zeros((batch, seq_len, num_value_heads), dtype=np.float32)
    b_values = np.zeros_like(a_values)
    q = _float32_to_bfloat16_bits(q_values)
    k = _float32_to_bfloat16_bits(k_values)
    v = _float32_to_bfloat16_bits(v_values)
    a = _float32_to_bfloat16_bits(a_values)
    b_gate = _float32_to_bfloat16_bits(b_values)
    a_log = np.zeros(num_value_heads, dtype=np.float32)
    dt_bias = np.zeros(num_value_heads, dtype=np.float32)
    scale = float(1.0 / math.sqrt(_HEAD_DIM))
    expected_output, expected_state = gdn_decode_bf16_numpy_reference(
        q,
        k,
        v,
        a,
        b_gate,
        a_log,
        dt_bias,
        scale=scale,
    )
    state_slot_stride = num_value_heads * _HEAD_DIM * _HEAD_DIM
    state_head_stride = _HEAD_DIM * _HEAD_DIM
    q_batch_stride = seq_len * num_heads * _HEAD_DIM
    v_batch_stride = seq_len * num_value_heads * _HEAD_DIM
    output_binding = np.zeros(batch * v_batch_stride, dtype=np.uint16)
    state_binding = np.zeros(batch * state_slot_stride, dtype=state_numpy_dtype)
    args: dict[str, Any] = {
        "state": state_binding,
        "intermediate": np.zeros(1, dtype=state_numpy_dtype),
        "A_log": a_log,
        "a": a.reshape(-1),
        "dt_bias": dt_bias,
        "q": q.reshape(-1),
        "k": k.reshape(-1),
        "v": v.reshape(-1),
        "b_gate": b_gate.reshape(-1),
        "output": output_binding,
        "read_indices": np.arange(batch, dtype=np.int32),
        "write_indices": np.arange(batch, dtype=np.int32),
        "state_slot_stride": state_slot_stride,
        "q_batch_stride": q_batch_stride,
        "k_batch_stride": q_batch_stride,
        "v_batch_stride": v_batch_stride,
    }
    parameter_names = {parameter.name for parameter in kernel.params}
    if "accepted_steps" in parameter_names:
        args["accepted_steps"] = np.zeros(1, dtype=np.int32)
    if "ssm_state_indices" in parameter_names:
        args["ssm_state_indices"] = np.zeros(1, dtype=np.int32)
    if "state_head_stride" in parameter_names:
        args["state_head_stride"] = state_head_stride
    return NumSimCase(
        kernel=kernel,
        args=args,
        outputs=("output", "state"),
        reference=lambda: {
            "output": expected_output.reshape(-1).copy(),
            "state": expected_state.reshape(-1).copy(),
        },
        comparisons={
            "output": ComparisonSpec(rtol=2e-2, atol=2e-3, actual_encoding="bfloat16"),
            "state": ComparisonSpec(
                rtol=2e-2,
                atol=2e-3,
                actual_encoding="bfloat16" if state_dtype == "bfloat16" else None,
            ),
        },
    )


def prepare_gdn_decode_ilp4_case() -> NumSimCase:
    return _prepare_gdn_decode_case(
        gdn_decode_bf16_ilp4,
        {
            "seq_len": 1,
            "batch": 1,
            "num_heads": 2,
            "num_v_heads": 4,
            "tile_v": 16,
            "use_qk_l2norm": True,
            "disable_state_update": False,
            "cache_intermediate_states": False,
            "disable_output": False,
        },
        seq_len=1,
    )


def prepare_gdn_decode_mtp_case() -> NumSimCase:
    return _prepare_gdn_decode_case(
        gdn_decode_bf16_wide_vec_mtp,
        {
            "seq_len": 2,
            "batch": 1,
            "num_heads": 16,
            "num_v_heads": 32,
            "tile_v": 32,
            "use_qk_l2norm": True,
            "disable_state_update": False,
            "cache_intermediate_states": False,
            "disable_output": False,
        },
        seq_len=2,
    )


def prepare_gdn_decode_t1_case() -> NumSimCase:
    return _prepare_gdn_decode_case(
        gdn_decode_bf16_wide_vec_t1,
        {
            "batch": 1,
            "num_heads": 2,
            "num_v_heads": 4,
            "tile_v": 64,
            "use_qk_l2norm": True,
            "disable_state_update": False,
            "cache_intermediate_states": False,
            "same_pool": True,
        },
        seq_len=1,
    )


def prepare_gdn_decode_fp32_mtp_warp_case() -> NumSimCase:
    """Exercise the smallest supported work-unit count without native offsets."""

    return _prepare_gdn_decode_case(
        gdn_decode_fp32_mtp_warp,
        {
            "seq_len": 3,
            "batch": 17,
            "num_heads": 2,
            "num_v_heads": 8,
            "tile_v": 32,
            "ilp_rows": 4,
            "use_smem_v": False,
            "use_qk_l2norm": True,
            "disable_state_update": False,
            "cache_intermediate_states": False,
            "same_pool": True,
        },
        seq_len=3,
        state_dtype="float32",
    )


def prepare_gdn_prefill_case() -> NumSimCase:
    """Exercise the smallest canonical HQ=2, HV=8 single-token GDN launch."""

    total_tokens = 1
    num_sequences = 1
    num_sms = 1
    num_q_heads = 2
    num_value_heads = 8
    seq_lens = (1,)
    q = np.zeros((total_tokens, num_q_heads, _HEAD_DIM), dtype=np.float16)
    q[..., 0] = np.float16(1.0)
    k = q.copy()
    v = np.zeros((total_tokens, num_value_heads, _HEAD_DIM), dtype=np.float16)
    base_vector = np.array([1, 2, 3, 4], dtype=np.float16)
    for head in range(num_value_heads):
        v[0, head, :4] = base_vector * np.float16((head + 1) / 16)
    gate = np.full((total_tokens, num_value_heads), np.float32(0.75), dtype=np.float32)
    beta = np.full((total_tokens, num_value_heads), np.float32(0.5), dtype=np.float32)
    output = np.zeros_like(v)
    initial_state = np.zeros(
        (num_sequences, num_value_heads, _HEAD_DIM, _HEAD_DIM), dtype=np.float32
    )
    final_state = np.zeros_like(initial_state)
    cu_seqlens = np.array([0, total_tokens], dtype=np.int32)
    scale = float(1.0 / math.sqrt(_HEAD_DIM))

    q_binding = q.reshape(-1)
    k_binding = k.reshape(-1)
    v_binding = v.reshape(-1)
    output_binding = output.reshape(-1)
    qk_shape = (_HEAD_DIM, total_tokens, num_q_heads)
    qk_strides = (2 * _HEAD_DIM * num_q_heads, 2 * _HEAD_DIM)
    value_heads_per_q_head = num_value_heads // num_q_heads
    vo_shape = (_HEAD_DIM, total_tokens, value_heads_per_q_head, num_q_heads)
    vo_strides = (
        2 * _HEAD_DIM * num_value_heads,
        2 * _HEAD_DIM,
        2 * _HEAD_DIM * value_heads_per_q_head,
    )
    box_shape_qk = (64, 64, 1)
    box_shape_vo = (64, 64, 1, 1)
    expected_output, expected_state = gdn_numpy_reference(
        q,
        k,
        v,
        gate,
        beta,
        initial_state,
        seq_lens=seq_lens,
        scale=scale,
    )
    kernel = _specialize_runtime_scalars(
        gdn_prefill_sm100.get_kernel(hq=num_q_heads, hv=num_value_heads, seq_lens=seq_lens),
        {
            "total_tokens": total_tokens,
            "num_sequences": num_sequences,
            "num_sms": num_sms,
        },
    )
    args = {
        "q": q_binding,
        "k": k_binding,
        "v": v_binding,
        "gate": gate.reshape(-1),
        "beta": beta.reshape(-1),
        "o": output_binding,
        "cu_seqlens": cu_seqlens,
        "initial_state": initial_state.reshape(-1),
        "final_state": final_state.reshape(-1),
        "q_map": _tensor_map(
            q_binding,
            dtype="float16",
            global_shape=qk_shape,
            global_strides=qk_strides,
            box_shape=box_shape_qk,
            swizzle="128B",
        ),
        "k_map": _tensor_map(
            k_binding,
            dtype="float16",
            global_shape=qk_shape,
            global_strides=qk_strides,
            box_shape=box_shape_qk,
            swizzle="128B",
        ),
        "v_map": _tensor_map(
            v_binding,
            dtype="float16",
            global_shape=vo_shape,
            global_strides=vo_strides,
            box_shape=box_shape_vo,
            swizzle="128B",
        ),
        "o_map": _tensor_map(
            output_binding,
            dtype="float16",
            global_shape=vo_shape,
            global_strides=vo_strides,
            box_shape=box_shape_vo,
            swizzle="128B",
        ),
        "desc_ws": np.zeros(num_sms * _GDN_DESCRIPTOR_BYTES_PER_CTA, dtype=np.int8),
        "scale": scale,
    }
    return NumSimCase(
        kernel=kernel,
        args=args,
        outputs=("o", "final_state"),
        reference=lambda: {
            "o": expected_output.reshape(-1).copy(),
            "final_state": expected_state.reshape(-1).copy(),
        },
        comparisons={
            "o": ComparisonSpec(rtol=2e-3, atol=2e-3),
            "final_state": ComparisonSpec(rtol=2e-3, atol=2e-3),
        },
    )


def prepare_gdn_cp_prefill_case() -> NumSimCase:
    """Exercise the four-launch CP chain across a chunk boundary and tail."""

    seq_lens = (65,)
    total_tokens = sum(seq_lens)
    num_sequences = 1
    num_heads = 1
    cp_chunk_len = 64
    total_chunks = 2
    total_t_blocks = 2
    scale = 1.0
    config = {
        "dtype": "float16",
        "q_heads": num_heads,
        "k_heads": num_heads,
        "v_heads": num_heads,
        "seq_lens": seq_lens,
        "cu_seqlens_dtype": "int32",
        "state_mode": "initial_final",
        "state_dtype": "float32",
        "indexed_state": False,
        "cp_chunk_len": cp_chunk_len,
        "gate_baseline": 0.75,
        "scale": scale,
        "seed": 17110,
    }

    q = np.zeros((total_tokens, num_heads, _HEAD_DIM), dtype=np.float16)
    q[..., 0] = np.float16(1.0)
    k = q.copy()
    v = np.zeros_like(q)
    base_vector = np.array([1, 2, 3, 4], dtype=np.float16)
    for token in range(total_tokens):
        v[token, 0, :4] = base_vector * np.float16((token + 1) / 512)
    alpha = np.full((total_tokens, num_heads), np.float32(0.75), dtype=np.float32)
    beta = np.full((total_tokens, num_heads), np.float32(0.5), dtype=np.float32)
    initial_state = np.zeros((num_sequences, num_heads, _HEAD_DIM, _HEAD_DIM), dtype=np.float32)
    initial_state[0, 0, 0, 0] = np.float32(0.25)
    final_state = np.zeros_like(initial_state)
    output = np.zeros_like(v)
    cu_seqlens = np.array([0, total_tokens], dtype=np.int32)
    state_indices = np.array([0], dtype=np.int32)

    t_workspace = np.zeros((total_t_blocks, num_heads, 64, 64), dtype=np.float16)
    cp_shape = (total_chunks, num_heads, _HEAD_DIM, _HEAD_DIM)
    transfer = np.zeros(cp_shape, dtype=np.float32)
    local_state = np.zeros(cp_shape, dtype=np.float32)
    fixed_state = np.zeros(cp_shape, dtype=np.float32)
    initial_state_workspace = np.zeros_like(initial_state)
    descriptor_workspace = np.zeros(
        num_sequences * num_heads * total_chunks * 5 * 128,
        dtype=np.int8,
    )

    q_binding = q.reshape(-1)
    k_binding = k.reshape(-1)
    v_binding = v.reshape(-1)
    alpha_binding = alpha.reshape(-1)
    beta_binding = beta.reshape(-1)
    t_binding = t_workspace.reshape(-1)
    transfer_binding = transfer.reshape(-1)
    local_state_binding = local_state.reshape(-1)
    fixed_state_binding = fixed_state.reshape(-1)
    initial_state_binding = initial_state.reshape(-1)
    final_state_binding = final_state.reshape(-1)
    initial_state_workspace_binding = initial_state_workspace.reshape(-1)
    output_binding = output.reshape(-1)
    cu_seqlens_binding = cu_seqlens
    state_indices_binding = state_indices
    descriptor_workspace_binding = descriptor_workspace

    element_bytes = 2
    qkv_shape = (_HEAD_DIM, total_tokens, num_heads)
    qkv_strides = (
        element_bytes * _HEAD_DIM * num_heads,
        element_bytes * _HEAD_DIM,
    )
    qkv_box = (64, 64, 1)
    mn_t_shape = (64, 64, num_heads, total_t_blocks)
    mn_t_strides = (
        element_bytes * 64,
        element_bytes * 64 * 64,
        element_bytes * 64 * 64 * num_heads,
    )
    prefill_t_shape = (64, 64, 1, num_heads, total_t_blocks)
    prefill_t_strides = (
        element_bytes * 64,
        element_bytes * 64 * 64,
        element_bytes * 64 * 64,
        element_bytes * 64 * 64 * num_heads,
    )
    output_shape = (_HEAD_DIM, total_tokens, 1, num_heads)
    output_strides = (
        element_bytes * _HEAD_DIM * num_heads,
        element_bytes * _HEAD_DIM,
        element_bytes * _HEAD_DIM,
    )

    kernels = gdn_cp_prefill_sm100.get_kernel(**config)
    kernel_chain = (
        kernels["t_precompute"],
        kernels["mn_precompute"],
        kernels["fixup_simt_row4"],
        kernels["prefill"],
    )
    args = {
        "k0:k": k_binding,
        "k0:beta": beta_binding,
        "k0:t": t_binding,
        "k0:cu_seqlens": cu_seqlens_binding,
        "k0:k_map": _tensor_map(
            k_binding,
            dtype="float16",
            global_shape=qkv_shape,
            global_strides=qkv_strides,
            box_shape=qkv_box,
            swizzle="128B",
        ),
        "k1:k": k_binding,
        "k1:v": v_binding,
        "k1:t": t_binding,
        "k1:alpha": alpha_binding,
        "k1:transfer": transfer_binding,
        "k1:local_state": local_state_binding,
        "k1:cu_seqlens": cu_seqlens_binding,
        "k1:k_map": _tensor_map(
            k_binding,
            dtype="float16",
            global_shape=qkv_shape,
            global_strides=qkv_strides,
            box_shape=qkv_box,
            swizzle="128B",
        ),
        "k1:v_map": _tensor_map(
            v_binding,
            dtype="float16",
            global_shape=qkv_shape,
            global_strides=qkv_strides,
            box_shape=qkv_box,
            swizzle="128B",
        ),
        "k1:t_map": _tensor_map(
            t_binding,
            dtype="float16",
            global_shape=mn_t_shape,
            global_strides=mn_t_strides,
            box_shape=(64, 64, 1, 1),
            swizzle="128B",
        ),
        "k2:transfer": transfer_binding,
        "k2:local_state": local_state_binding,
        "k2:initial_state": initial_state_binding,
        "k2:initial_state_workspace": initial_state_workspace_binding,
        "k2:fixed_state": fixed_state_binding,
        "k2:final_state": final_state_binding,
        "k2:state_indices": state_indices_binding,
        "k2:cu_seqlens": cu_seqlens_binding,
        "k3:q": q_binding,
        "k3:k": k_binding,
        "k3:v": v_binding,
        "k3:alpha": alpha_binding,
        "k3:t": t_binding,
        "k3:fixed_state": fixed_state_binding,
        "k3:initial_state_workspace": initial_state_workspace_binding,
        "k3:o": output_binding,
        "k3:cu_seqlens": cu_seqlens_binding,
        "k3:scale": scale,
        "k3:q_map": _tensor_map(
            q_binding,
            dtype="float16",
            global_shape=output_shape,
            global_strides=output_strides,
            box_shape=(64, 64, 1, 1),
            swizzle="128B",
        ),
        "k3:k_map": _tensor_map(
            k_binding,
            dtype="float16",
            global_shape=qkv_shape,
            global_strides=qkv_strides,
            box_shape=qkv_box,
            swizzle="128B",
        ),
        "k3:v_map": _tensor_map(
            v_binding,
            dtype="float16",
            global_shape=qkv_shape,
            global_strides=qkv_strides,
            box_shape=qkv_box,
            swizzle="128B",
        ),
        "k3:t_map": _tensor_map(
            t_binding,
            dtype="float16",
            global_shape=prefill_t_shape,
            global_strides=prefill_t_strides,
            box_shape=(64, 64, 1, 1, 1),
            swizzle="128B",
        ),
        "k3:o_map": _tensor_map(
            output_binding,
            dtype="float16",
            global_shape=output_shape,
            global_strides=output_strides,
            box_shape=(64, 64, 1, 1),
            swizzle="128B",
        ),
        "k3:descriptor_workspace": descriptor_workspace_binding,
    }
    expected_output, expected_state = gdn_numpy_reference(
        q,
        k,
        v,
        alpha,
        beta,
        initial_state,
        seq_lens=seq_lens,
        scale=scale,
    )
    return NumSimCase(
        kernel=kernel_chain,
        args=args,
        outputs=("k2:final_state", "k3:o"),
        reference=lambda: {
            "k2:final_state": expected_state.reshape(-1).copy(),
            "k3:o": expected_output.reshape(-1).copy(),
        },
        comparisons={
            "k2:final_state": ComparisonSpec(rtol=1e-2, atol=1e-2),
            "k3:o": ComparisonSpec(rtol=8e-3, atol=8e-3),
        },
    )


__all__ = [
    "gdn_decode_bf16_numpy_reference",
    "gdn_numpy_reference",
    "kda_decode_numpy_reference",
    "prepare_native_kda_fixed_case",
    "prepare_native_kda_forward_case",
    "prepare_gdn_decode_ilp4_case",
    "prepare_gdn_decode_mtp_case",
    "prepare_gdn_decode_fp32_mtp_warp_case",
    "prepare_gdn_decode_t1_case",
    "prepare_gdn_cp_prefill_case",
    "prepare_gdn_prefill_case",
    "prepare_recurrent_kda_grouped_case",
    "prepare_recurrent_kda_one_warp_case",
]
