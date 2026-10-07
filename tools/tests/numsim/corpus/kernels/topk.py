"""CPU-owned, non-degenerate cases for the canonical FlashInfer Top-K family."""

from __future__ import annotations

import numpy as np
from tvm import tirx
from tvm_ffi import structural_map

from tests.numsim.support._tirx_kernels import load_tirx_kernel
from tirx_harness.numsim.cases import ComparisonSpec, NumSimCase


def _specialize(kernel, values: dict[str, int | float]):
    return kernel.specialize(
        {param: values[param.name] for param in kernel.params if param.name in values}
    )


def _scores(length: int) -> np.ndarray:
    """Distinct, monotone scores make both selected values and indices observable."""

    return np.arange(length, dtype=np.float32) / np.float32(length)


def _radix_multi_cta_state_oracle(module, length: int) -> np.ndarray:
    """Return the post-collect row state for the adapter's single group.

    The output counter is a protocol result: one claim is made per CTA in the
    group.  The canonical module owns the workspace ABI offsets; deriving the
    group size from its launch arithmetic keeps this oracle independent of the
    kernel's implementation statements while avoiding a layout index literal.
    """

    max_chunk = module.max_chunk_elements("float32", length, False, False)
    ctas_per_group = (length + max_chunk - 1) // max_chunk
    if ctas_per_group < 2:
        raise AssertionError("state oracle requires the multi-CTA dispatch")
    output_counter_word = module.OUTPUT_COUNTER_WORD
    if output_counter_word >= module.ROW_STATE_WORDS:
        raise AssertionError("canonical row-state output counter is out of bounds")
    state = np.zeros((module.WORKSPACE_WORDS,), dtype=np.uint32)
    state[output_counter_word] = ctas_per_group
    return state


def _name_buffer_parameters(kernel, names: tuple[str, ...]):
    """Give an anonymous public Buffer ABI stable adapter-local names."""

    if len(names) != len(kernel.params):
        raise ValueError(f"expected {len(kernel.params)} parameter names, got {len(names)}")
    replacements = {}
    params = []
    for parameter, name in zip(kernel.params, names, strict=True):
        if parameter.name:
            if parameter.name != name:
                raise ValueError(f"kernel parameter {parameter.name!r} is not {name!r}")
            params.append(parameter)
            continue
        renamed = tirx.Var(name, parameter.ty, parameter.span)
        replacements[parameter] = renamed
        params.append(renamed)
    if not replacements:
        return kernel
    return tirx.PrimFunc(
        params,
        structural_map(kernel.body, (tirx.Var, lambda var: replacements.get(var, var)), order="post"),
        kernel.ret_type,
        kernel.attrs,
        kernel.span,
    )


def _topk_case(name: str, *, length: int, k: int, expected_order: str) -> NumSimCase:
    module = load_tirx_kernel(name)
    scores = _scores(length)
    indices = np.full((k,), -1, dtype=np.int32)
    values = np.full((k,), np.nan, dtype=np.float32)
    expected_indices = np.arange(length - k, length, dtype=np.int32)
    if expected_order == "descending":
        expected_indices = expected_indices[::-1].copy()
    expected_values = scores[expected_indices]
    if name == "radix_topk_single_cta":
        kernel = module.get_kernel(num_rows=1, length=length, k=k, mode="basic", dtype="float32")
        aux = np.zeros((1,), dtype=np.int32)
        row_lengths = np.full((1,), length, dtype=np.int32)
        zeros = np.zeros((1,), dtype=np.int32)
        args = {
            "inp": scores,
            "out_idx": indices,
            "out_val": values,
            "aux": aux,
            "lengths_g": row_lengths,
            "row_starts_g": zeros,
            "pt_starts_g": zeros,
            "row_to_batch_g": zeros,
            "aux_stride": 1,
        }
        outputs = ("out_idx", "out_val")
        expected = {"out_idx": expected_indices, "out_val": expected_values}
    elif name == "radix_topk_multi_cta":
        kernel = module.get_kernel(num_rows=1, length=length, k=k, mode="basic", dtype="float32")
        aux = np.zeros((1,), dtype=np.int32)
        row_lengths = np.full((1,), length, dtype=np.int32)
        zeros = np.zeros((1,), dtype=np.int32)
        state = np.zeros((module.WORKSPACE_WORDS,), dtype=np.uint32)
        args = {
            "inp": scores,
            "out_idx": indices,
            "out_val": values,
            "aux": aux,
            "lengths_g": row_lengths,
            "row_starts_g": zeros,
            "pt_starts_g": zeros,
            "row_to_batch_g": zeros,
            "state": state,
            "aux_stride": 1,
        }
        # The multi-CTA protocol clears its per-group state during the exit
        # phase only after the final CTA's arrival.  This two-CTA dispatch
        # leaves the output counter at two; retain it as an observed output so
        # the complete protocol state is independently checked.
        outputs = ("out_idx", "out_val", "state")
        expected_state = _radix_multi_cta_state_oracle(module, length)
        expected = {
            "out_idx": expected_indices,
            "out_val": expected_values,
            "state": expected_state,
        }
    elif name == "filtered_topk":
        kernel = module.get_kernel(
            dtype="float32", mode="basic", num_rows=1, length=length, k=k, pattern="random"
        )
        aux = np.zeros((1,), dtype=np.int32)
        row_lengths = np.full((1,), length, dtype=np.int32)
        zeros = np.zeros((1,), dtype=np.int32)
        args = {
            "inp": scores,
            "out_idx": indices,
            "out_val": values,
            "aux": aux,
            "lengths_g": row_lengths,
            "row_starts_g": zeros,
            "pt_starts_g": zeros,
            "row_to_batch_g": zeros,
            "aux_stride": 1,
        }
        outputs = ("out_idx", "out_val")
        expected = {"out_idx": expected_indices, "out_val": expected_values}
    else:
        raise ValueError(f"unsupported Top-K case {name}")
    return NumSimCase(
        kernel=kernel,
        args=args,
        outputs=outputs,
        reference=lambda expected=expected: {key: value.copy() for key, value in expected.items()},
        comparisons={key: ComparisonSpec(rtol=0, atol=0) for key in outputs},
    )


def prepare_filtered_topk_case() -> NumSimCase:
    return _topk_case("filtered_topk", length=1024, k=2, expected_order="ascending")


def prepare_radix_topk_single_cta_case() -> NumSimCase:
    return _topk_case("radix_topk_single_cta", length=1024, k=2, expected_order="descending")


def prepare_radix_topk_multi_cta_case() -> NumSimCase:
    return _topk_case("radix_topk_multi_cta", length=65536, k=2, expected_order="descending")


def prepare_fast_topk_clusters_case() -> NumSimCase:
    module = load_tirx_kernel("fast_topk_clusters")
    # The source launcher only reaches this exact cluster path for rows at
    # least one short-row threshold; a 1024-element row takes a different
    # production dispatch and leaves the staged histogram incomplete.
    length, k = 8192, 2
    scores = _scores(length)
    indices = np.full((k,), -1, dtype=np.int32)
    values = np.full((k,), np.nan, dtype=np.float32)
    overflow = np.zeros((4 * length,), dtype=np.int32)
    expected_indices = np.arange(length - k, length, dtype=np.int32)
    args = {
        "logits": scores,
        "indices": indices,
        "values": values,
        "overflow": overflow,
    }
    kernel = module.get_kernel(dtype="float32", batch=1, seq_len=length, k=k, mode="plain")
    kernel = _name_buffer_parameters(kernel, ("logits", "indices", "values", "overflow"))
    # The production path only uses overflow as a spill buffer.  With k=2 and
    # this strictly ordered input no candidate spills, so its complete mutated
    # state remains the independently initialized zero buffer.
    expected = {
        "indices": expected_indices,
        "values": scores[expected_indices],
        "overflow": np.zeros_like(overflow),
    }
    return NumSimCase(
        kernel=kernel,
        args=args,
        outputs=("indices", "values", "overflow"),
        reference=lambda: {key: value.copy() for key, value in expected.items()},
        comparisons={
            "indices": ComparisonSpec(rtol=0, atol=0),
            "values": ComparisonSpec(rtol=0, atol=0),
        },
    )


def prepare_stable_sort_topk_by_value_case() -> NumSimCase:
    module = load_tirx_kernel("stable_sort_topk_by_value")
    # Equal values are the observable part of this kernel's contract: the
    # radix sort must preserve the input order of the satellite indices.
    values = np.array([0.125, 0.125], dtype=np.float32)
    indices = np.array([11, 7], dtype=np.int32)
    original_values = values.copy()
    original_indices = indices.copy()
    kernel = module.get_kernel(dtype="float32", num_rows=1, k=2, pattern="tie_heavy")
    expected_order = np.argsort(-original_values, kind="stable")
    expected = {
        "out_idx": original_indices[expected_order],
        "out_val": original_values[expected_order],
    }
    return NumSimCase(
        kernel=kernel,
        args={
            "out_idx": indices,
            "out_val": values,
        },
        outputs=("out_idx", "out_val"),
        reference=lambda: {key: value.copy() for key, value in expected.items()},
        comparisons={
            "out_idx": ComparisonSpec(rtol=0, atol=0),
            "out_val": ComparisonSpec(rtol=0, atol=0),
        },
    )


__all__ = [
    "prepare_fast_topk_clusters_case",
    "prepare_filtered_topk_case",
    "prepare_radix_topk_multi_cta_case",
    "prepare_radix_topk_single_cta_case",
    "prepare_stable_sort_topk_by_value_case",
]
