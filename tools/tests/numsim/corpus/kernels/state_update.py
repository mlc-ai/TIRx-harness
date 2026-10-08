"""CUDA-free NumSim cases for canonical selective-state-update kernels."""

from __future__ import annotations

from typing import Any

import numpy as np

from tests.numsim.corpus.kernels.recurrent import (
    _bfloat16_bits_to_float32,
    _float32_to_bfloat16_bits,
)
from tests.numsim.support._tirx_kernels import load_tirx_kernel
from tirx_harness.numsim.cases import (
    ComparisonSpec,
    NumSimCase,
    TensorMap,
)
from tirx_harness.numsim.host_abi import build_host_abi
from tirx_harness.numsim.transpiler.frontend import analyze

selective_state_update_stp_simple = load_tirx_kernel("selective_state_update_stp_simple")
selective_state_update_stp_vertical = load_tirx_kernel("selective_state_update_stp_vertical")
selective_state_update_stp_horizontal = load_tirx_kernel("selective_state_update_stp_horizontal")
selective_state_update_mtp_simple = load_tirx_kernel("selective_state_update_mtp_simple")
selective_state_update_mtp_vertical = load_tirx_kernel("selective_state_update_mtp_vertical")
selective_state_update_mtp_horizontal = load_tirx_kernel("selective_state_update_mtp_horizontal")

_BATCH = 1
_NHEADS = 1
_DIM = 64
_DSTATE = 64
_NGROUPS = 1
_MTP_TOKENS = 2
_STATE_STRIDE = _NHEADS * _DIM * _DSTATE


def selective_state_update_numpy_reference(
    state_bits: np.ndarray,
    x_bits: np.ndarray,
    dt: np.ndarray,
    matrix_a: np.ndarray,
    matrix_b_bits: np.ndarray,
    matrix_c_bits: np.ndarray,
    d_weight: np.ndarray,
    state_indices: np.ndarray,
    *,
    heads_per_group: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Token-serial FP32 oracle for the fixed, unscaled selective recurrence."""

    state = _bfloat16_bits_to_float32(state_bits).copy()
    x = _bfloat16_bits_to_float32(x_bits)
    matrix_b = _bfloat16_bits_to_float32(matrix_b_bits)
    matrix_c = _bfloat16_bits_to_float32(matrix_c_bits)
    batch, tokens, nheads, dim = x.shape
    output = np.zeros_like(x, dtype=np.float32)

    for batch_index in range(batch):
        state_slot = int(state_indices[batch_index])
        for token in range(tokens):
            for head in range(nheads):
                group = head // heads_per_group
                dt_value = np.float32(dt[batch_index, token, head])
                decay = np.float32(np.exp(np.float32(matrix_a[head]) * dt_value))
                b_row = matrix_b[batch_index, token, group]
                c_row = matrix_c[batch_index, token, group]
                for row in range(dim):
                    x_value = np.float32(x[batch_index, token, head, row])
                    current = (
                        state[state_slot, head, row] * decay + b_row * (dt_value * x_value)
                    ).astype(np.float32)
                    state[state_slot, head, row] = current
                    output[batch_index, token, head, row] = np.float32(
                        np.sum(current * c_row, dtype=np.float32)
                        + np.float32(d_weight[head]) * x_value
                    )
    return output, state


def _config(*, tokens: int) -> dict[str, Any]:
    config: dict[str, Any] = {
        "label": "numsim_b1_h1_d64_s64",
        "batch": _BATCH,
        "nheads": _NHEADS,
        "dim": _DIM,
        "dstate": _DSTATE,
        "input_dtype": "bfloat16",
        "state_dtype": "bfloat16",
        "weight_dtype": "float32",
        "matrix_a_dtype": "float32",
        "index_dtype": "int64",
        "index_rank": 1,
        "has_state_indices": True,
        "has_dst_indices": False,
        "has_z": False,
        "has_d": True,
        "has_dt_bias": True,
        "dt_softplus": False,
        "update_state": True,
        "state_stride_factor": 1,
        "pad_every": 0,
        "use_out_tensor": True,
        "philox_rounds": 0,
        "seed": 0,
    }
    if tokens == 1:
        config["ngroups"] = _NGROUPS
    else:
        config.update(
            {
                "tokens": tokens,
                "heads_per_group": _NHEADS // _NGROUPS,
                "cu_seqlens_dtype": "int32",
                "accepted_dtype": "int64",
                "mode": "fixed",
                "has_intermediate_states": False,
                "has_num_accepted_tokens": False,
                "shared_state_slot": False,
            }
        )
    return config


def _tensor_map(
    base: np.ndarray,
    *,
    global_shape: tuple[int, ...],
    global_strides: tuple[int, ...],
    box_shape: tuple[int, ...],
) -> np.ndarray:
    return TensorMap(
        base=base,
        dtype="bfloat16",
        global_shape=global_shape,
        global_strides=global_strides,
        box_shape=box_shape,
        element_strides=(1,) * len(global_shape),
        swizzle=None,
        interleave=None,
        fill_mode="none",
    ).numpy()


def _prepare_selective_state_update_case(
    module: Any,
    *,
    tokens: int,
    schedule: str,
) -> NumSimCase:
    config = _config(tokens=tokens)
    kernel = module.get_kernel(**config)
    if schedule == "stp_simple":
        dim_tiles = _DIM // 4
        substitutions = {
            parameter: dim_tiles
            for parameter in kernel.params
            if parameter.name == "dim_tiles_runtime"
        }
        kernel = kernel.specialize(substitutions)
    state_slots = _BATCH

    x_values = np.empty((_BATCH, tokens, _NHEADS, _DIM), dtype=np.float32)
    row_values = np.arange(1, _DIM + 1, dtype=np.float32) / np.float32(256.0)
    for token in range(tokens):
        x_values[:, token, :, :] = row_values * np.float32(token + 1)
    matrix_b_values = np.zeros((_BATCH, tokens, _NGROUPS, _DSTATE), dtype=np.float32)
    matrix_c_values = np.zeros_like(matrix_b_values)
    matrix_b_values[..., 0] = np.float32(1.0)
    matrix_c_values[..., 0] = np.float32(1.0)

    state_bits = np.zeros((state_slots, _NHEADS, _DIM, _DSTATE), dtype=np.uint16)
    x_bits = _float32_to_bfloat16_bits(x_values)
    matrix_b_bits = _float32_to_bfloat16_bits(matrix_b_values)
    matrix_c_bits = _float32_to_bfloat16_bits(matrix_c_values)
    dt = np.full((_BATCH, tokens, _NHEADS), np.float32(0.5), dtype=np.float32)
    matrix_a = np.zeros(_NHEADS, dtype=np.float32)
    d_weight = np.full(_NHEADS, np.float32(0.25), dtype=np.float32)
    state_indices = np.zeros(_BATCH, dtype=np.int64)
    expected_output, expected_state = selective_state_update_numpy_reference(
        state_bits,
        x_bits,
        dt,
        matrix_a,
        matrix_b_bits,
        matrix_c_bits,
        d_weight,
        state_indices,
        heads_per_group=_NHEADS // _NGROUPS,
    )

    state_binding = state_bits.reshape(-1)
    x_binding = x_bits.reshape(-1)
    matrix_b_binding = matrix_b_bits.reshape(-1)
    matrix_c_binding = matrix_c_bits.reshape(-1)
    output_binding = np.zeros(x_bits.size, dtype=np.uint16)
    float_placeholder = np.zeros(1, dtype=np.float32)
    index_placeholder = np.zeros(1, dtype=np.int64)
    int32_placeholder = np.array([0, tokens], dtype=np.int32)
    state_scale_binding = np.zeros(1, dtype=np.float32)
    intermediate_binding = np.zeros(1, dtype=np.uint16)

    candidates: dict[str, Any] = {
        "state": state_binding,
        "state_scale": state_scale_binding,
        "state_scale_h": state_scale_binding,
        "x": x_binding,
        "dt": dt.reshape(-1),
        "matrix_a": matrix_a,
        "matrix_b": matrix_b_binding,
        "matrix_c": matrix_c_binding,
        "d_weight": d_weight,
        "z": x_binding,
        "dt_bias": float_placeholder,
        "state_indices": state_indices,
        "dst_indices": index_placeholder,
        "dst_indices_h": index_placeholder,
        "intermediate_states": intermediate_binding,
        "intermediate_indices": index_placeholder,
        "intermediate_scales": float_placeholder,
        "intermediate_scales_h": float_placeholder,
        "cu_seqlens": int32_placeholder,
        "cu_seqlens_h": int32_placeholder,
        "num_accepted_tokens": index_placeholder,
        "num_accepted_tokens_h": index_placeholder,
        "rand_seed": index_placeholder,
        "output": output_binding,
        "state_stride_batch": _STATE_STRIDE,
        "state_scale_stride_batch": 0,
        "x_stride_batch": tokens * _NHEADS * _DIM,
        "x_stride_mtp": _NHEADS * _DIM,
        "dt_stride_batch": tokens * _NHEADS,
        "dt_stride_mtp": _NHEADS,
        "b_stride_batch": tokens * _NGROUPS * _DSTATE,
        "b_stride_mtp": _NGROUPS * _DSTATE,
        "c_stride_batch": tokens * _NGROUPS * _DSTATE,
        "c_stride_mtp": _NGROUPS * _DSTATE,
        "z_stride_batch": tokens * _NHEADS * _DIM,
        "z_stride_mtp": _NHEADS * _DIM,
        "out_stride_batch": tokens * _NHEADS * _DIM,
        "out_stride_mtp": _NHEADS * _DIM,
        "state_indices_stride_batch": 1,
        "state_indices_stride_t": 0,
        "dst_indices_stride_batch": 0,
        "dst_indices_stride_t": 0,
        "cache_steps": tokens,
        "nheads_runtime": _NHEADS,
        "ngroups_runtime": _NGROUPS,
        "dt_softplus": 0,
        "update_state": 1,
        "pad_slot_id": -1,
        "dim_tiles_runtime": 16,
    }

    state_shape = (_DSTATE, _DIM, _NHEADS, state_slots)
    state_strides = (
        _DSTATE * 2,
        _DSTATE * _DIM * 2,
        _STATE_STRIDE * 2,
    )
    if schedule == "stp_vertical":
        candidates["tensor_state"] = _tensor_map(
            state_binding,
            global_shape=state_shape,
            global_strides=state_strides,
            box_shape=(_DSTATE, 16, 1, 1),
        )
    elif schedule == "stp_horizontal":
        candidates["tensor_state"] = _tensor_map(
            state_binding,
            global_shape=state_shape,
            global_strides=state_strides,
            box_shape=(32, _DIM, 1, 1),
        )
    elif schedule in {"mtp_vertical", "mtp_horizontal"}:
        state_box = (_DSTATE, _DIM, 1, 1) if schedule == "mtp_vertical" else (_DSTATE, 32, 1, 1)
        candidates["tensor_state"] = _tensor_map(
            state_binding,
            global_shape=state_shape,
            global_strides=state_strides,
            box_shape=state_box,
        )
        bc_shape = (_DSTATE, _NGROUPS, tokens, _BATCH)
        bc_strides = (
            _DSTATE * 2,
            _NGROUPS * _DSTATE * 2,
            tokens * _NGROUPS * _DSTATE * 2,
        )
        candidates["tensor_b"] = _tensor_map(
            matrix_b_binding,
            global_shape=bc_shape,
            global_strides=bc_strides,
            box_shape=(_DSTATE, 1, tokens, 1),
        )
        candidates["tensor_c"] = _tensor_map(
            matrix_c_binding,
            global_shape=bc_shape,
            global_strides=bc_strides,
            box_shape=(_DSTATE, 1, tokens, 1),
        )
        candidates["tensor_x"] = _tensor_map(
            x_binding,
            global_shape=(_DIM, _NHEADS, tokens, _BATCH),
            global_strides=(
                _DIM * 2,
                _NHEADS * _DIM * 2,
                tokens * _NHEADS * _DIM * 2,
            ),
            box_shape=(_DIM, 1, tokens, 1),
        )

    module_spec = analyze(kernel)
    host_abi = build_host_abi(module_spec)
    binding_names = {
        slot.canonical_name
        for slot in host_abi.slots
        if slot.canonical_name not in host_abi.implicit_tensor_map_names
    }
    missing = binding_names - candidates.keys()
    if missing:
        raise ValueError(f"missing selective-state-update bindings: {sorted(missing)}")
    args = {name: candidates[name] for name in binding_names}
    return NumSimCase(
        kernel=kernel,
        args=args,
        outputs=("output", "state"),
        reference=lambda: {
            "output": expected_output.reshape(-1).copy(),
            "state": expected_state.reshape(-1).copy(),
        },
        comparisons={
            "output": ComparisonSpec(rtol=1e-2, atol=2e-3, actual_encoding="bfloat16"),
            "state": ComparisonSpec(rtol=1e-2, atol=2e-3, actual_encoding="bfloat16"),
        },
    )


def prepare_selective_state_update_stp_simple_case() -> NumSimCase:
    return _prepare_selective_state_update_case(
        selective_state_update_stp_simple, tokens=1, schedule="stp_simple"
    )


def prepare_selective_state_update_stp_vertical_case() -> NumSimCase:
    return _prepare_selective_state_update_case(
        selective_state_update_stp_vertical, tokens=1, schedule="stp_vertical"
    )


def prepare_selective_state_update_stp_horizontal_case() -> NumSimCase:
    return _prepare_selective_state_update_case(
        selective_state_update_stp_horizontal,
        tokens=1,
        schedule="stp_horizontal",
    )


def prepare_selective_state_update_mtp_simple_case() -> NumSimCase:
    return _prepare_selective_state_update_case(
        selective_state_update_mtp_simple, tokens=_MTP_TOKENS, schedule="mtp_simple"
    )


def prepare_selective_state_update_mtp_vertical_case() -> NumSimCase:
    return _prepare_selective_state_update_case(
        selective_state_update_mtp_vertical,
        tokens=_MTP_TOKENS,
        schedule="mtp_vertical",
    )


def prepare_selective_state_update_mtp_horizontal_case() -> NumSimCase:
    return _prepare_selective_state_update_case(
        selective_state_update_mtp_horizontal,
        tokens=_MTP_TOKENS,
        schedule="mtp_horizontal",
    )


__all__ = [
    "prepare_selective_state_update_mtp_horizontal_case",
    "prepare_selective_state_update_mtp_simple_case",
    "prepare_selective_state_update_mtp_vertical_case",
    "prepare_selective_state_update_stp_horizontal_case",
    "prepare_selective_state_update_stp_simple_case",
    "prepare_selective_state_update_stp_vertical_case",
    "selective_state_update_numpy_reference",
]
