"""NumSim-sized expert-parallel runs of the ported DeepGEMM ``mega_moe``.

Every rank owns ``experts_per_rank`` routed experts and one shared expert, and
routes its tokens to experts on any rank. ``symm_buffer`` is a
:class:`~tirx_harness.numsim.SymmetricBuffer`, and the kernel reaches the other
ranks' replicas through ``symm_rank_offset_<peer>``: the byte distance from this
rank's replica to the peer's, as the GPU launcher computes it from the
symmetric-memory handle.

The reference is the TP1 corpus oracle evaluated on the experts of all ranks.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any
from unittest.mock import patch

import numpy as np

from ported.deepgemm import sm100_fp8_fp4_mega_moe as module
from ported.deepgemm._sm100_fp8_fp4_mega_moe import spec
from tests.numsim.corpus.kernels.deepgemm import (
    _align_up,
    _mega_moe_interleave_l1_rows,
    _mega_moe_reference,
    _mega_moe_scale_storage,
    _pack_e2m1,
    _pack_e8m0_words,
)
from tirx_harness.numsim import SymmetricBuffer
from tirx_harness.numsim.cases import TensorMap

MAX_RANKS = 72
FP8_CODES = np.array([0x20, 0x28, 0x30, 0x34, 0x38, 0xA0, 0xA8, 0xB0, 0xB4, 0xB8], dtype=np.uint8)
FP4_CODES = np.array([0x1, 0x2, 0x3, 0x9, 0xA, 0xB], dtype=np.uint8)


@dataclass(frozen=True)
class Shape:
    world: int = 4
    num_tokens: tuple[int, ...] = (2, 2, 2, 2)
    num_max_tokens_per_rank: int = 4
    hidden: int = 256
    intermediate_hidden: int = 256
    experts_per_rank: int = 2
    num_topk: int = 2
    num_shared_experts: int = 1
    activation_clamp: float = 1.0
    num_sms: int = 4

    def config(self) -> Any:
        return module.MegaMoeConfig(
            num_processes=self.world,
            num_max_tokens_per_rank=self.num_max_tokens_per_rank,
            num_tokens=max(self.num_tokens),
            hidden=self.hidden,
            intermediate_hidden=self.intermediate_hidden,
            num_experts=self.experts_per_rank * self.world,
            num_topk=self.num_topk,
            num_shared_experts=self.num_shared_experts,
            activation_clamp=self.activation_clamp,
            fast_math=1,
        )


def _environment(shape: Shape):
    return patch.dict(os.environ, {"TIRX_DEEPGEMM_NUM_SMS_OVERRIDE": str(shape.num_sms)})


def kernel(shape: Shape = Shape()):
    config = shape.config()
    with _environment(shape):
        return module.get_kernel(
            num_processes=config.num_processes,
            num_max_tokens_per_rank=config.num_max_tokens_per_rank,
            num_tokens=config.num_tokens,
            hidden=config.hidden,
            intermediate_hidden=config.intermediate_hidden,
            num_experts=config.num_experts,
            num_topk=config.num_topk,
            num_shared_experts=config.num_shared_experts,
            activation_clamp=config.activation_clamp,
            fast_math=config.fast_math,
            collect_stats=True,
        )


def _aligned(array: np.ndarray) -> np.ndarray:
    # 16U4_ALIGN16B TensorMaps need a 32-byte aligned global address.
    owner = np.empty(array.nbytes + 31, dtype=np.uint8)
    offset = -owner.ctypes.data % 32
    result = owner[offset : offset + array.nbytes].view(array.dtype).reshape(array.shape)
    np.copyto(result, array)
    return result


def _tensor_map(base, *, dtype, global_shape, global_strides, box_shape, swizzle, fp4=False):
    return TensorMap(
        base=base,
        dtype="float4_e2m1fn" if fp4 else dtype,
        global_shape=global_shape,
        global_strides=global_strides,
        box_shape=box_shape,
        element_strides=(1,) * len(global_shape),
        fp4_shared_layout="align16_padded" if fp4 else None,
        swizzle=swizzle,
        interleave=None,
        fill_mode="none",
    ).numpy()


def _rank_data(shape: Shape, rank: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
    config = shape.config()
    tokens = shape.num_tokens[rank]
    hidden, inter = config.hidden, config.intermediate_hidden
    shared = config.shared_intermediate_hidden
    routes = np.stack([rng.permutation(config.num_experts)[: config.num_topk]
                       for _ in range(tokens)]).astype(np.int64).reshape(tokens, config.num_topk)
    return {
        "input_codes": rng.choice(FP8_CODES, size=(tokens, hidden)).astype(np.uint8),
        "input_scale_exponents": rng.integers(-1, 2, size=(tokens, hidden // 32), dtype=np.int16),
        "routes": routes,
        "route_weights": rng.choice(np.float32([0.25, 0.5, 0.75, 1.25]),
                                    size=routes.shape).astype(np.float32),
        "l1_codes": rng.choice(FP4_CODES, size=(shape.experts_per_rank, inter * 2, hidden)),
        "l2_codes": rng.choice(FP4_CODES, size=(shape.experts_per_rank, hidden, inter)),
        "l1_scale_exponents": rng.integers(
            -3, 0, size=(shape.experts_per_rank, inter * 2, hidden // 32), dtype=np.int16),
        "l2_scale_exponents": rng.integers(
            -4, -1, size=(shape.experts_per_rank, hidden, inter // 32), dtype=np.int16),
        "shared_l1_codes": rng.choice(FP8_CODES, size=(shared * 2, hidden)).astype(np.uint8),
        "shared_l2_codes": rng.choice(FP8_CODES, size=(hidden, shared)).astype(np.uint8),
        "shared_l1_scale_exponents": rng.integers(
            -4, -1, size=(shared * 2, hidden // 32), dtype=np.int16),
        "shared_l2_scale_exponents": rng.integers(
            -4, -1, size=(hidden, shared // 32), dtype=np.int16),
        "stats": (np.arange(shape.experts_per_rank, dtype=np.int32) + 7 + 10 * rank),
    }


def _rank_args(shape: Shape, rank: int, data: dict[str, np.ndarray]) -> dict[str, Any]:
    """Everything one rank binds except the peer offsets."""

    config = shape.config()
    with _environment(shape):
        launch = spec.get_deepgemm_launch_config(config)
        workspace = spec.get_deepgemm_workspace_layout(config)
        layout = spec.get_deepgemm_symm_buffer_layout(config)
    hidden, inter = config.hidden, config.intermediate_hidden
    shared = config.shared_intermediate_hidden
    tokens = shape.num_tokens[rank]
    max_tokens = workspace.num_max_tokens_per_rank

    symm = np.zeros((layout.total_bytes,), dtype=np.int8)
    raw = symm.view(np.uint8)

    def region(offset: int, size: int) -> np.ndarray:
        return raw[offset : offset + size]

    region(layout.input_token_offset, max_tokens * hidden).reshape(max_tokens, hidden)[
        :tokens] = data["input_codes"]
    region(layout.input_sf_offset, max_tokens * (hidden // 32)).reshape(
        max_tokens, hidden // 32)[:tokens] = (data["input_scale_exponents"] + 127).astype(np.uint8)
    region(layout.input_topk_idx_offset, max_tokens * config.num_topk * 8).view(np.int64).reshape(
        max_tokens, config.num_topk)[:tokens] = data["routes"]
    region(layout.input_topk_weights_offset, max_tokens * config.num_topk * 4).view(
        np.float32).reshape(max_tokens, config.num_topk)[:tokens] = data["route_weights"]

    shared_l1_sf_size = layout.num_max_shared_sf_tokens * (hidden // 32)
    shared_l1_sf = region(layout.shared_l1_sf_offset, shared_l1_sf_size).view(np.uint32).reshape(
        hidden // 128, layout.num_max_shared_sf_tokens)
    words = _pack_e8m0_words(data["input_scale_exponents"].reshape(tokens, -1, 4))
    for token in range(tokens):
        in_block = token % launch.block_m
        shared_l1_sf[:, (in_block // 128) * 128 + (in_block % 32) * 4 + (in_block % 128) // 32] = (
            words[token])

    l1_weights = _aligned(_mega_moe_interleave_l1_rows(_pack_e2m1(data["l1_codes"])))
    l2_weights = _aligned(_pack_e2m1(data["l2_codes"]))
    l1_weight_sf = _aligned(_mega_moe_scale_storage(data["l1_scale_exponents"], interleave_l1=True))
    l2_weight_sf = _aligned(_mega_moe_scale_storage(data["l2_scale_exponents"],
                                                    interleave_l1=False))
    shared_l1_weights = _aligned(_mega_moe_interleave_l1_rows(data["shared_l1_codes"][None])[0])
    shared_l2_weights = _aligned(data["shared_l2_codes"])
    shared_l1_weight_sf = _aligned(_mega_moe_scale_storage(
        data["shared_l1_scale_exponents"][None], interleave_l1=True))
    shared_l2_weight_sf = _aligned(_mega_moe_scale_storage(
        data["shared_l2_scale_exponents"][None], interleave_l1=False))

    sf_block_m = _align_up(launch.block_m, 128)
    ring, sf_ring = workspace.num_ring_tokens, workspace.num_sf_ring_tokens
    shared_sf = layout.num_max_shared_sf_tokens
    fp8, i32 = "float8_e4m3fn", "int32"
    maps = {
        "tensor_map_l1_acts": _tensor_map(
            region(layout.l1_token_offset, ring * hidden), dtype=fp8,
            global_shape=(hidden, ring), global_strides=(hidden,),
            box_shape=(128, launch.load_block_m), swizzle="128B"),
        "tensor_map_l1_acts_sf": _tensor_map(
            region(layout.l1_sf_offset, sf_ring * (hidden // 32)).view(np.int32), dtype=i32,
            global_shape=(sf_ring, hidden // 128), global_strides=(sf_ring * 4,),
            box_shape=(sf_block_m, launch.block_k // 128), swizzle=None),
        "tensor_map_l1_weights": _tensor_map(
            l1_weights, dtype="uint8",
            global_shape=(hidden, shape.experts_per_rank * inter * 2), global_strides=(hidden // 2,),
            box_shape=(128, launch.load_block_n), swizzle="128B", fp4=True),
        "tensor_map_l1_weights_sf": _tensor_map(
            l1_weight_sf, dtype=i32,
            global_shape=(inter * 2, shape.experts_per_rank * (hidden // 128)),
            global_strides=(inter * 2 * 4,),
            box_shape=(launch.block_n, launch.block_k // 128), swizzle=None),
        "tensor_map_l1_output": _tensor_map(
            region(layout.l2_token_offset, ring * inter), dtype=fp8,
            global_shape=(inter, ring), global_strides=(inter,),
            box_shape=(64, launch.store_block_m), swizzle="64B"),
        "tensor_map_l2_acts": _tensor_map(
            region(layout.l2_token_offset, ring * inter), dtype=fp8,
            global_shape=(inter, ring), global_strides=(inter,),
            box_shape=(128, launch.load_block_m), swizzle="128B"),
        "tensor_map_l2_acts_sf": _tensor_map(
            region(layout.l2_sf_offset, sf_ring * (inter // 32)).view(np.int32), dtype=i32,
            global_shape=(sf_ring, inter // 128), global_strides=(sf_ring * 4,),
            box_shape=(sf_block_m, launch.block_k // 128), swizzle=None),
        "tensor_map_l2_weights": _tensor_map(
            l2_weights, dtype="uint8",
            global_shape=(inter, shape.experts_per_rank * hidden), global_strides=(inter // 2,),
            box_shape=(128, launch.load_block_n), swizzle="128B", fp4=True),
        "tensor_map_l2_weights_sf": _tensor_map(
            l2_weight_sf, dtype=i32,
            global_shape=(hidden, shape.experts_per_rank * (inter // 128)),
            global_strides=(hidden * 4,),
            box_shape=(launch.block_n, launch.block_k // 128), swizzle=None),
        "tensor_map_shared_l1_acts": _tensor_map(
            region(layout.shared_l1_token_offset, max_tokens * hidden), dtype=fp8,
            global_shape=(hidden, max_tokens), global_strides=(hidden,),
            box_shape=(128, launch.load_block_m), swizzle="128B"),
        "tensor_map_shared_l1_acts_sf": _tensor_map(
            region(layout.shared_l1_sf_offset, shared_l1_sf_size).view(np.int32), dtype=i32,
            global_shape=(shared_sf, hidden // 128), global_strides=(shared_sf * 4,),
            box_shape=(sf_block_m, launch.block_k // 128), swizzle=None),
        "tensor_map_shared_l1_weights": _tensor_map(
            shared_l1_weights, dtype="uint8",
            global_shape=(hidden, shared * 2), global_strides=(hidden,),
            box_shape=(128, launch.load_block_n), swizzle="128B"),
        "tensor_map_shared_l1_weights_sf": _tensor_map(
            shared_l1_weight_sf, dtype=i32,
            global_shape=(shared * 2, hidden // 128), global_strides=(shared * 2 * 4,),
            box_shape=(launch.block_n, launch.block_k // 128), swizzle=None),
        "tensor_map_shared_l1_output": _tensor_map(
            region(layout.shared_l2_token_offset, max_tokens * shared), dtype=fp8,
            global_shape=(shared, max_tokens), global_strides=(shared,),
            box_shape=(64, launch.store_block_m), swizzle="64B"),
        "tensor_map_shared_l2_acts": _tensor_map(
            region(layout.shared_l2_token_offset, max_tokens * shared), dtype=fp8,
            global_shape=(shared, max_tokens), global_strides=(shared,),
            box_shape=(128, launch.load_block_m), swizzle="128B"),
        "tensor_map_shared_l2_acts_sf": _tensor_map(
            region(layout.shared_l2_sf_offset, shared_sf * (shared // 32)).view(np.int32),
            dtype=i32, global_shape=(shared_sf, shared // 128), global_strides=(shared_sf * 4,),
            box_shape=(sf_block_m, launch.block_k // 128), swizzle=None),
        "tensor_map_shared_l2_weights": _tensor_map(
            shared_l2_weights, dtype="uint8",
            global_shape=(shared, hidden), global_strides=(shared,),
            box_shape=(128, launch.load_block_n), swizzle="128B"),
        "tensor_map_shared_l2_weights_sf": _tensor_map(
            shared_l2_weight_sf, dtype=i32,
            global_shape=(hidden, shared // 128), global_strides=(hidden * 4,),
            box_shape=(launch.block_n, launch.block_k // 128), swizzle=None),
    }
    return {
        "y": np.full((tokens * hidden,), np.uint16(0x7FC0), dtype=np.uint16),
        "cumulative_local_expert_recv_stats": data["stats"].copy(),
        "symm_buffer": symm,
        **maps,
        "num_tokens": np.int32(tokens),
        "rank_idx": np.int32(rank),
    }


def rank_inputs(shape: Shape = Shape(), *, seed: int = 4800) -> tuple[list[dict], list[dict]]:
    """Per-rank bindings and each rank's expected ``y`` and expert statistics."""

    if shape.num_shared_experts != 1:
        raise ValueError("the expert-parallel NumSim case covers one shared expert")
    if len(shape.num_tokens) != shape.world:
        raise ValueError("num_tokens needs one entry per rank")
    rng = np.random.default_rng(seed)
    data = [_rank_data(shape, rank, rng) for rank in range(shape.world)]
    rows = [_rank_args(shape, rank, data[rank]) for rank in range(shape.world)]
    symm = SymmetricBuffer([row["symm_buffer"] for row in rows])
    for rank, row in enumerate(rows):
        offsets = symm.peer_offsets(rank)
        row["symm_buffer"] = symm
        row.update({f"symm_rank_offset_{peer}": offsets[peer] if peer < shape.world
                    else np.int64(0) for peer in range(MAX_RANKS)})

    every = {name: np.concatenate([d[name] for d in data]) for name in
             ("l1_codes", "l2_codes", "l1_scale_exponents", "l2_scale_exponents")}
    received = np.bincount(np.concatenate([d["routes"].ravel() for d in data]),
                           minlength=shape.experts_per_rank * shape.world)
    expected = []
    for rank, d in enumerate(data):
        y = _mega_moe_reference(
            input_codes=d["input_codes"], input_scale_exponents=d["input_scale_exponents"],
            l1_packed=_pack_e2m1(every["l1_codes"]), l1_scale_exponents=every["l1_scale_exponents"],
            l2_packed=_pack_e2m1(every["l2_codes"]), l2_scale_exponents=every["l2_scale_exponents"],
            topk_idx=d["routes"], topk_weights=d["route_weights"],
            activation_clamp=shape.activation_clamp,
            shared_l1_codes=d["shared_l1_codes"],
            shared_l1_scale_exponents=d["shared_l1_scale_exponents"],
            shared_l2_codes=d["shared_l2_codes"],
            shared_l2_scale_exponents=d["shared_l2_scale_exponents"])
        own = slice(rank * shape.experts_per_rank, (rank + 1) * shape.experts_per_rank)
        expected.append({"y": y.reshape(-1),
                         "cumulative_local_expert_recv_stats": d["stats"] + received[own]})
    return rows, expected
