"""Descriptor validation applies only when the MMA actually issues."""

import numpy as np
import pytest
from tvm_ffi import structural_map

from tests.numsim.runtime.test_tcgen05_ti16 import ti16_kernel
from tests.numsim.support.execution import assert_rejected, run_checked
from tirx_harness.numsim.transpiler.ptx_dialect import decode_ptx_call
from tvm import tirx
from tvm.script import tirx as T


# Distinct instruction dispatch paths, not a Cartesian modifier matrix.
BOUNDARY_CASES = (
    ("mixed_ss", "f16", False, False, False, 0),
    ("mixed_ts", "f16", True, False, False, 0),
    ("mixed_ws", "f16", False, True, False, 0),
    ("mixed_sparse", "f16", False, False, True, 0),
    ("tf32_ts", "tf32", True, False, False, 15),
    ("i8_ts", "i8", True, False, False, 15),
    ("f8_ts", "f8f6f4", True, False, False, 15),
    ("f16_sparse_ts", "f16", True, False, True, 15),
    ("ti16_transpose_a", "ti16", False, False, False, 15),
    ("ti16_transpose_b", "ti16", False, False, False, 16),
)


def inactive_boundary_case(case):
    name, kind, tmem, ws, sparse, transpose_bit = case
    kernel = ti16_kernel(
        True,
        tmem,
        ws=ws,
        kind=kind,
        sparse=sparse,
        b_format=int(name.startswith("mixed")),
        collectors=".collector::a::discard.collector::b::discard" if sparse else "",
        arch="sm_107a" if sparse or kind == "ti16" else "sm_100a",
    )
    if transpose_bit:

        def set_transpose(node):
            call = node.value
            if type(call).__name__ != "Call" or not str(call.op.name).startswith(
                "tirx.ptx.tcgen05_mma"
            ):
                return node
            descriptor = decode_ptx_call(call).scalar_operand("idesc")
            args = [
                T.uint32(int(arg.value) | (1 << transpose_bit)) if arg.same_as(descriptor) else arg
                for arg in call.args
            ]
            return tirx.Evaluate(
                type(call)(
                    call.op,
                    args,
                    attrs=call.attrs,
                    ty_args=call.ty_args,
                    span=call.span,
                    ret_ty=call.ty,
                )
            )

        kernel = kernel.with_body(structural_map(kernel.body, (tirx.Evaluate, set_transpose)))
    columns = 64 if ws else 16
    inputs = {
        "a": np.zeros((4, 128, 16), np.uint16),
        "b": np.zeros((96 if ws else 16, 32 if sparse else 16), np.uint16),
        "metadata": np.zeros((2, 128, 2), np.uint32),
        "seed": np.zeros((128, columns), np.int32),
        "out": np.zeros((128, columns), np.int32),
        "zero_mask": np.zeros(1, np.uint64),
    }
    inputs["seed"][:] = np.arange(inputs["seed"].size, dtype=np.int32).reshape(inputs["seed"].shape)
    inputs["zero_mask"][0] = np.uint64(1 << 63)
    reason = (
        "requires matching F16/BF16"
        if name.startswith("mixed")
        else "ti16_transpose_unmodeled"
        if kind == "ti16"
        else "TMEM A must be K-major"
    )
    return kernel, inputs, reason


@pytest.mark.parametrize("case", BOUNDARY_CASES, ids=[case[0] for case in BOUNDARY_CASES])
def test_tcgen_boundary_follows_instruction_predicate(case):
    kernel, inputs, reason = inactive_boundary_case(case)
    actual = run_checked(kernel, inputs, outputs=("out",))
    np.testing.assert_array_equal(actual.outputs["out"], inputs["seed"])
    inputs["zero_mask"][0] = 0
    reports = assert_rejected(
        kernel, inputs, reason, verdict="incomplete" if case[1] == "ti16" else "error"
    )
    for report in reports:
        assert any(finding.details.get("operation") for finding in report.findings), report.format()
