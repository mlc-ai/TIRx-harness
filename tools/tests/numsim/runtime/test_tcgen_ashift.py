"""Post-MMA A shifting uses the same issue and completion as the matrix product."""

import numpy as np
import pytest
from tvm import tirx
from tvm_ffi import structural_map
from tvm.script import tirx as T

from tests.numsim.runtime.test_tcgen05_ti16 import ti16_kernel
from tests.numsim.support.execution import assert_rejected, run_checked
from tirx_harness.numsim.transpiler.ptx_dialect import decode_ptx_call


def ashift_case(
    cta_group=1,
    *,
    followup=True,
    kind="f16",
    sparse=False,
    bf16=False,
    m=None,
):
    m, n = m or 128 * cta_group, 16 * cta_group
    banks = 2 if cta_group == 2 and m == 128 and not sparse else 1
    kernel = ti16_kernel(
        False,
        tmem_a=True,
        cta_group=cta_group,
        m=m,
        kind=kind,
        sparse=sparse,
        collectors=".collector::a::discard.collector::b::discard"
        if sparse and kind in {"tf32", "f16", "f8f6f4"}
        else "",
        a_format=int(bf16),
        b_format=int(bf16),
        arch="sm_107a",
        sparsity_selector=0 if kind == "f8f6f4" else 1,
    )

    def replace_mma(node):
        if not isinstance(node, tirx.Evaluate) or type(node.value).__name__ != "Call":
            return node
        if not str(node.value.op.name).startswith("tirx.ptx.tcgen05_mma"):
            return node
        decoded = decode_ptx_call(node.value)
        names = ("d_tmem", "a_tmem", "b_desc", *(("sp_meta_tmem",) if sparse else ()), "idesc")
        operands = [decoded.scalar_operand(name) for name in names]
        shifted = T.ptx[
            f"tcgen05.mma{'.sp' if sparse else ''}.cta_group::{cta_group}.kind::{kind}.ashift.collector::b::discard"
        ](
            *operands,
            *decoded.operand("disable_output_lane"),
            T.ptx.pred(T.uint32(0)),
            pred=True,
        )
        return tirx.SeqStmt([tirx.Evaluate(shifted), node]) if followup else tirx.Evaluate(shifted)

    kernel = kernel.with_body(structural_map(kernel.body, (tirx.Evaluate, replace_mma)))
    a_k = 32 if kind == "f8f6f4" else 8 if kind == "tf32" else 16
    b_k = a_k * (2 if sparse else 1)
    a = (np.arange(4 * m * a_k).reshape(4, m, a_k) % 113 - 56).astype(np.float32)
    if kind == "f8f6f4":
        a = (a % 7 - 3).astype(np.float32)
    b = (np.arange(n * b_k).reshape(n, b_k) % 7 - 3).astype(np.float32)
    shifted_a = a[:banks].copy()
    if followup:
        for start in range(0, m, 32):
            shifted_a[:, start : start + 31] = a[:banks, start + 1 : start + 32]
    if sparse:
        expanded = np.zeros((banks, m, b_k), np.float32)
        if kind == "tf32":
            expanded[:, :, 0::2] = shifted_a
        else:
            expanded[:, :, 0::4], expanded[:, :, 1::4] = (
                shifted_a[:, :, 0::2],
                shifted_a[:, :, 1::2],
            )
        shifted_a = expanded
    expected = np.concatenate(
        [
            shifted_a[bank] @ b[bank * (n // banks) : (bank + 1) * (n // banks)].T
            for bank in range(banks)
        ],
        axis=1,
    )
    if cta_group == 2:
        expected[m // 2 + 1, : n // banks] = 0

    def encode(value):
        if kind == "f8f6f4":
            # Exact E4M3 encodings of 0,1,2,3, with the independent sign bit.
            bits = np.array([0, 0x38, 0x40, 0x44], np.uint8)[np.abs(value).astype(np.int64)]
            return (bits | ((value < 0).astype(np.uint8) << 7)).view(np.uint16)
        if kind == "ti16":
            return np.abs(value).astype(np.uint16) | ((value < 0).astype(np.uint16) << 15)
        if kind == "tf32":
            return value.view(np.uint16)
        if bf16:
            return (value.view(np.uint32) >> 16).astype(np.uint16)
        return value.astype(np.float16).view(np.uint16)

    metadata = np.zeros((2, 128, 4 if sparse and a_k == 64 else 2), np.uint32)
    if sparse:
        metadata[:, :, 1] = 0x44444444  # 2:4 keeps positions 0/1; TF32 1:2 keeps position 0.
        if kind == "f8f6f4":
            metadata.fill(0x44444444)
    return (
        kernel,
        {
            "a": encode(a),
            "b": encode(b),
            "zero_mask": np.zeros(1, np.uint64),
            "metadata": metadata,
            "seed": np.zeros((m, n), np.int32),
            "out": np.zeros((m, n), np.int32),
        },
        expected.astype(np.int32) if kind == "ti16" else expected.view(np.int32),
    )


@pytest.mark.parametrize(
    "cta_group,kind,sparse,bf16",
    [
        (1, "f16", False, False),
        (2, "f16", False, False),
        (1, "f16", False, True),
        (1, "tf32", False, False),
        (2, "ti16", False, False),
        (1, "ti16", True, False),
        (2, "ti16", True, False),
        (1, "tf32", True, False),
        (2, "tf32", True, False),
        (1, "f16", True, False),
        (2, "f16", True, True),
        (1, "f8f6f4", True, False),
        (2, "f8f6f4", True, False),
    ],
)
def test_ashift_followed_by_mma(cta_group, kind, sparse, bf16, tmp_path):
    for followup in (False, True):
        kernel, args, expected = ashift_case(
            cta_group, followup=followup, kind=kind, sparse=sparse, bf16=bf16
        )
        outputs = run_checked(kernel, args, cache_dir=tmp_path).outputs
        np.testing.assert_array_equal(outputs["out"], expected)


@pytest.mark.parametrize("kind", ("f16", "tf32", "f8f6f4"))
def test_sparse_m128_ashift_reports_issuing_operation(kind):
    # Use the existing collector-B entry for this CPU-side diagnostic contract.
    kernel, inputs, _ = ashift_case(2, m=128, kind=kind, sparse=True, followup=False)
    reports = assert_rejected(
        kernel, inputs, "tcgen_sparse_m128_ashift_unmodeled", verdict="incomplete"
    )
    for report in reports:
        operation = report.findings[0].details.get("operation")
        assert operation, report.format()
        assert operation["source_op_id"] == operation["source"]["source_op_id"]
        assert "T.ptx.tcgen05" in operation["source"]["source_text"]
