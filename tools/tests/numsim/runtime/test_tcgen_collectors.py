"""Collector validity and ordinary operand reads share the existing MMA path."""

import numpy as np
import pytest
from tvm_ffi import structural_map

from tests.numsim.runtime.test_tcgen05_ti16 import ti16_kernel
from tests.numsim.support.execution import assert_rejected, run_checked
from tirx_harness.numsim.transpiler.ptx_dialect import decode_ptx_call
from tvm_tirx import ir
from tvm_tirx.script import tirx as T


def collector_case(sequence, *, tmem_a=False, ws=False, slot="a", cta_group=1):
    m = 128 * cta_group
    n = 64 if ws else 16 * cta_group
    b_rows = n + 32 if ws else n
    kind = "ti16" if ws else "f16"
    kernel = ti16_kernel(
        False,
        tmem_a,
        cta_group=cta_group,
        m=m,
        ws=ws,
        kind=kind,
        arch="sm_107a" if ws or tmem_a or slot == "b" else "sm_100a",
    )

    def replace_mma(node):
        if not isinstance(node, ir.Evaluate) or type(node.value).__name__ != "Call":
            return node
        if not str(node.value.op.name).startswith("tirx.ptx.tcgen05_mma"):
            return node
        decoded = decode_ptx_call(node.value)
        prefix = f"tcgen05.mma{'.ws' if ws else ''}.cta_group::{cta_group}.kind::{kind}"
        instructions = []
        enabled = 0
        for action in sequence:
            execute = not action.startswith("!")
            action = action.removeprefix("!")
            opcode = prefix + (f".collector::{slot}::{action}" if action else "")
            # The current TVM TS collector-A entry requires explicit collector B.
            if action and tmem_a and not ws and slot == "a":
                opcode += ".collector::b::discard"
            if action and slot == "b" and not ws:
                opcode = prefix + f".collector::a::discard.collector::b::{action}"
            operands = [
                decoded.scalar_operand("d_tmem"),
                decoded.scalar_operand("a_tmem" if tmem_a else "a_desc"),
                decoded.scalar_operand("b_desc"),
                decoded.scalar_operand("idesc"),
            ]
            if not ws:
                operands.extend(decoded.operand("disable_output_lane"))
            operands.append(T.ptx.pred(T.uint32(enabled != 0)))
            if ws:
                operands.append(decoded.scalar_operand("zero_col_mask"))
            call = T.ptx[opcode](*operands, pred=execute)
            instructions.append(ir.Evaluate(call))
            enabled += execute
        return instructions[0] if len(instructions) == 1 else ir.SeqStmt(instructions)

    kernel = kernel.with_body(structural_map(kernel.body, (ir.Evaluate, replace_mma)))
    a = np.arange(4 * m * 16).reshape(4, m, 16) % 7 - 3
    b = np.arange(b_rows * 16).reshape(b_rows, 16) % 5 - 2
    expected = (a[0].astype(np.float32) @ b[:n].astype(np.float32).T) * sum(
        not action.startswith("!") for action in sequence
    )
    if cta_group == 2:
        expected[129] = 0  # Existing helper's disabled physical output lane.
    encode = (
        (lambda value: (np.abs(value) | ((value < 0).astype(np.int64) << 15)).astype(np.uint16))
        if ws
        else (lambda value: value.astype(np.float16).view(np.uint16))
    )
    expected = expected.astype(np.int32) if ws else expected.astype(np.float32).view(np.int32)
    return (
        kernel,
        {
            "a": encode(a),
            "b": encode(b),
            "metadata": np.zeros((2, 128, 2), np.uint32),
            "zero_mask": np.zeros(1, np.uint64),
            "seed": np.zeros((m, n), np.int32),
            "out": np.zeros((m, n), np.int32),
        },
        expected,
    )


@pytest.mark.parametrize(
    "tmem_a,ws,slot,cta_group",
    [
        (False, False, "a", 1),
        (False, False, "b", 1),
        (True, False, "a", 2),
        (False, True, "b2", 1),
    ],
)
def test_collector_chain_and_false_discard(tmem_a, ws, slot, cta_group, tmp_path):
    kernel, args, expected = collector_case(
        ("fill", "!discard", "use", "lastuse", ""),
        tmem_a=tmem_a,
        ws=ws,
        slot=slot,
        cta_group=cta_group,
    )
    outputs = run_checked(kernel, args, cache_dir=tmp_path).outputs
    np.testing.assert_array_equal(outputs["out"], expected)


@pytest.mark.parametrize("sequence", [("use",), ("fill", "", "use"), ("fill", "lastuse", "use")])
def test_collector_use_requires_live_fill(sequence, tmp_path):
    kernel, args, _ = collector_case(sequence)
    assert_rejected(kernel, args, "requires a valid previous fill", cache_dir=tmp_path)


@pytest.mark.numsim_gpu
def test_collector_chain_gpu_oracle(gpu_runner):
    for options in ({}, {"cta_group": 2}):
        kernel, args, expected = collector_case(
            ("fill", "!discard", "use", "lastuse", ""), **options
        )
        outputs = gpu_runner(kernel, args, outputs=("out",), arch="sm_100a")
        np.testing.assert_array_equal(outputs["out"], expected)
