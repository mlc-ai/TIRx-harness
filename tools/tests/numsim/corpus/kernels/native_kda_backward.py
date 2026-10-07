"""A complete KDA backward launch with an independent one-token gradient oracle."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from tests.numsim.support._tirx_kernels import load_tirx_kernel
from tirx_harness.numsim.cases import ComparisonSpec, NumSimCase, TensorMap

from .recurrent import _float32_to_bfloat16_bits, _specialize_runtime_scalars


def prepare_native_kda_backward_case() -> NumSimCase:
    module = load_tirx_kernel("kda_backward_packed")
    dim, chunk = module.D, module.CHUNK
    scale = 1.0 / np.sqrt(dim)

    def vector(value):
        result = np.zeros(dim, dtype=np.float32)
        result[0] = value
        return result

    def state(value):
        result = np.zeros((dim, dim), dtype=np.float32)
        result[0, 0] = value
        return result

    q, k, v = vector(1.0), vector(1.0), vector(0.75)
    beta = np.array([0.5], dtype=np.float32)
    g = np.full(dim, -1.0, dtype=np.float32)
    h0, do, dht = state(0.125), vector(0.5), state(0.25)

    # The prepared g input stores log2 gates; FLA returns dg with respect to the
    # natural-log gate increments. Differentiate that mathematical recurrence,
    # independently of the saved interaction matrices consumed by the kernel.
    natural_gate = g.astype(np.float64) * np.log(2.0)
    leaves = [
        torch.tensor(value, dtype=torch.float64, requires_grad=True)
        for value in (q, k, v, beta, natural_gate, h0)
    ]
    tq, tk, tv, tb, tg, th = leaves
    decayed = tg.exp()[:, None] * th
    updated = decayed + tk[:, None] * (tb * (tv - tk @ decayed))[None, :]
    loss = ((tq @ updated) * scale * torch.tensor(do)).sum()
    loss = loss + (updated * torch.tensor(dht)).sum()
    gradients = torch.autograd.grad(loss, leaves)
    names = ("dq", "dk", "dv", "db", "dg", "dh0")
    expected = {
        name: value.detach().numpy().astype(np.float32).reshape(-1)
        for name, value in zip(names, gradients, strict=True)
    }

    # One token has no strict-lower-triangular interactions: Akk is the identity
    # and Aqk contains only its causal diagonal q.k * scale.
    aqk = np.zeros(chunk, dtype=np.float32)
    aqk[0] = np.dot(q, k) * scale
    akk = np.zeros(chunk, dtype=np.float32)
    akk[0] = 1.0
    args = {
        name: _float32_to_bfloat16_bits(value)
        for name, value in {
            "q": q,
            "k": k,
            "v": v,
            "beta": beta,
            "aqk": aqk,
            "akk": akk,
            "do": do,
        }.items()
    }
    args.update(g=g, h0=h0.reshape(-1), dht=dht.reshape(-1))
    args.update(
        egcache=np.zeros(dim, dtype=np.uint16),
        hsnap=np.zeros(2 * dim * dim, dtype=np.uint16),
        dhsnap=np.zeros(2 * dim * dim, dtype=np.uint16),
        cu_seqlens=np.array([0, 1], dtype=np.int64),
        stream_counter=np.zeros(4, dtype=np.int32),
        flags=np.zeros(2, dtype=np.int64),
        scale=np.float32(scale),
        num_seqs=np.int32(1),
        num_items=np.int32(1),
        epoch=np.int32(1),
        # The native launch runs a range guard and a native fast path before
        # the TIRx fallback. The guard marks this one-token range as needing
        # the exact diagonal path (bit 0), with no unsafe-range bit set.
        range_flags=np.array([1, 0, 0, 0], dtype=np.int32),
        range_allowed=np.int32(1),
        range_entries=np.int32(1),
    )
    for name, table in zip(
        ("stream_tab", "item_tab", "seq_tab"), module.build_mega_tables([1], 1, 1, 3), strict=True
    ):
        args[name] = table.numpy()
    for name, reference in expected.items():
        args[name] = np.full(
            reference.shape,
            0x7FC0 if name == "dv" else np.nan,
            dtype=np.uint16 if name == "dv" else np.float32,
        )

    def encode(tensor, dtype, dims, strides_bytes, box, swizzle=3, l2promo=2):
        return SimpleNamespace(
            ptr=TensorMap(
                base=tensor.numpy(),
                dtype=dtype,
                global_shape=tuple(dims),
                global_strides=tuple(strides_bytes),
                box_shape=tuple(box),
                element_strides=(1,) * len(dims),
                swizzle={0: None, 1: "32B", 2: "64B", 3: "128B"}[swizzle],
            ).numpy()
        )

    # Reuse production descriptor construction with a CPU descriptor carrier.
    with patch.object(module, "encode_tensor_map", encode):
        for name in ("q", "k", "v", "do", "egcache"):
            args[("eg" if name == "egcache" else name) + "_map"] = module.token_map(
                torch.from_numpy(args[name]), 1, 1, dim, 64
            ).ptr
        args["g_map"] = module.token_map(torch.from_numpy(g), 1, 1, dim, 32).ptr
        for name in ("aqk", "akk"):
            args[name + "_map"] = module.token_map(
                torch.from_numpy(args[name]), 1, 1, chunk, chunk
            ).ptr
        for name, backing in (("h_map", "hsnap"), ("dh_map", "dhsnap")):
            args[name] = module.state_map(torch.from_numpy(args[backing]), 2).ptr
    return NumSimCase(
        kernel=_specialize_runtime_scalars(
            module.get_kernel(num_qk_heads=1, num_v_heads=1, seq_lens=(1,))[-1],
            {"num_ctas": 3},
        ),
        args=args,
        outputs=names,
        reference=lambda: expected,
        comparisons={
            name: ComparisonSpec(
                rtol=5e-3, atol=1e-5, actual_encoding="bfloat16" if name == "dv" else None
            )
            for name in names
        },
    )
