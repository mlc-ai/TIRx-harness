"""Small routed inputs for the canonical FP8 MoE megakernels."""

from __future__ import annotations


import ml_dtypes
import numpy as np

from tests.numsim.support._tirx_kernels import load_tirx_kernel
from tirx_harness.numsim.cases import ComparisonSpec, NumSimCase, TensorMap


def prepare_native_alphamoe_case() -> NumSimCase:
    module = load_tirx_kernel("alphamoe_fp8_blockscale_qwen3next")
    # Use the native kernel's supported eight-token grid with two routes per
    # token, distinct expert scales, and distinct output-block scales.
    tokens, experts, topk = 8, 4, 2
    hidden, inter, block = module.HIDDEN, module.INTERMEDIATE, module.BLOCK_SIZE
    fp8, bf16 = ml_dtypes.float8_e4m3fn, ml_dtypes.bfloat16
    routes = np.tile(np.array([[0, 1], [1, 0]], dtype=np.int32), (tokens // 2, 1))
    route_weights = np.tile(
        np.array([[0.25, 0.75], [0.625, 0.375]], dtype=np.float32), (tokens // 2, 1)
    )
    rsf = 1.5
    x = np.full((tokens, hidden), 0.125, dtype=bf16)
    x[1::2] = 0.25
    w1 = np.full((experts, 2 * inter, hidden), 0.0625, dtype=fp8)
    w1s = np.full((experts, 2 * inter // block, hidden // block), 0.125, dtype=np.float32)
    w1s[:, inter // block :, :] = 0.25
    w1s[1] *= 0.5
    w2 = np.full((experts, hidden, inter), 0.015625, dtype=fp8)
    w2s = np.full((experts, hidden // block, inter // block), 0.125, dtype=np.float32)
    w2s[:, 1::2, :] = 0.25
    def tensor_map(base, global_shape, global_strides, box_shape):
        return TensorMap(
            base=base,
            dtype="float8_e4m3fn",
            global_shape=global_shape,
            global_strides=global_strides,
            box_shape=box_shape,
            element_strides=(1,) * len(global_shape),
            swizzle="128B",
            fill_mode="none",
        ).numpy()

    args = {
        "topk_ids": routes.ravel(),
        "topk_w": route_weights.ravel(),
        "hidden": x.view(np.int32).ravel(),
        "w1s": w1s.ravel(),
        "w2s": w2s.ravel(),
        "out": np.zeros(tokens * topk * (hidden // 2), dtype=np.uint64),
        "final_out": np.full((tokens, hidden), np.nan, dtype=bf16).view(np.uint16).ravel(),
        "sync_ctr": np.zeros(128, dtype=np.uint32),
        "xq_g": np.zeros(tokens * hidden // 4, dtype=np.uint32),
        "xs_g": np.zeros(tokens * (hidden // block), dtype=np.float32),
        "xflag_g": np.zeros(tokens, dtype=np.uint32),
        "tm_w1h": tensor_map(
            w1, (hidden, inter, 2 * experts), (hidden, inter * hidden),
            (module.BK, module.CH, 2),
        ),
        "tm_w2": tensor_map(
            w2, (inter, experts * hidden), (inter,), (module.BK, module.BM),
        ),
        "rsf": rsf,
        "epoch": np.uint32(1),
    }
    # Constant blocks survive group FP8 quantization: 448 * (value / 448).
    # Compute both projections independently, rounding each route contribution
    # and each route-ordered accumulation to BF16, as required by the kernel.
    expected = np.zeros((tokens, hidden), dtype=bf16)
    for token in range(tokens):
        for rank, expert in enumerate(routes[token]):
            gate, up = (
                hidden * float(x[token, 0]) * 0.0625 * w1s[expert, :, 0].astype(np.float64)
            )
            activation = gate * up / (1 + np.exp(-gate))
            scales = np.repeat(w2s[expert, :, 0].astype(np.float64), block)
            contribution = (
                activation * inter * 0.015625 * scales * route_weights[token, rank] * rsf
            ).astype(bf16)
            expected[token] = (
                expected[token].astype(np.float32) + contribution.astype(np.float32)
            ).astype(bf16)
    return NumSimCase(
        kernel=module.build_kernel(80, tokens, topk, experts, hidden, inter).func,
        args=args,
        outputs=("final_out",),
        reference=lambda: {"final_out": expected.view(np.uint16).ravel()},
        comparisons={"final_out": ComparisonSpec(rtol=0, atol=0)},
    )
