"""One MMA site selects existing operand codecs from its runtime descriptor."""

import numpy as np
import pytest
from tvm import tirx
from tvm.ir import Expr
from tvm_ffi import structural_map, structural_walk
from tvm.script import tirx as T
from tvm.tirx import Stmt

from tests.numsim.microtests.harness import require_numsim_gpu, run_gpu_primfunc
from tests.numsim.runtime.test_tcgen_collectors import collector_case
from tests.numsim.support.execution import assert_rejected, run_checked
from tests.numsim.support.tcgen_descriptor import INSTR_DESC, encode_dense_instr_descriptor_fields
from tirx_harness.numsim.transpiler.ptx_dialect import decode_ptx_call


def dispatch_case(
    *, bf16=False, enabled=True, multiple_issuers=False, input_descriptor=False,
    descriptor_xor=0, tmem_a=False,
):
    kernel, inputs, expected = collector_case(("",), tmem_a=tmem_a)
    kernel = kernel.with_attr("tirx.cuda_arch", "sm_100a")
    metadata = next(parameter for parameter in kernel.params if str(parameter) == "metadata")
    selector = tirx.BufferLoad(metadata, [0, 0, 0])
    if multiple_issuers:
        lanes = []
        structural_walk(
            kernel.body,
            lambda node: lanes.append(node)
            if type(node).__name__ == "Var" and str(node) == "lane"
            else None,
        )
        # Distinct types must not split one invalid issue into valid calls.
        selector = (
            tirx.BufferLoad(metadata, [0, lanes[0], 0])
            if input_descriptor
            else T.Cast("uint32", lanes[0])
        )
    predicate = tirx.NE(tirx.BufferLoad(metadata, [0, 0, 1]), T.uint32(0))
    descriptor_bits = 0

    def replace(node):
        nonlocal descriptor_bits
        if isinstance(node, tirx.Evaluate) and type(node.value).__name__ == "Call":
            if not str(node.value.op.name).startswith("tirx.ptx.tcgen05_mma"):
                return node
            decoded = decode_ptx_call(node.value)
            descriptor = int(decoded.scalar_operand("idesc").value)
            descriptor_bits = descriptor
            return tirx.Evaluate(
                T.ptx["tcgen05.mma.cta_group::1.kind::f16"](
                    decoded.scalar_operand("d_tmem"),
                    decoded.scalar_operand("a_tmem" if tmem_a else "a_desc"),
                    decoded.scalar_operand("b_desc"),
                    selector if input_descriptor else T.Select(
                        tirx.NE(selector, T.uint32(0)),
                        T.uint32(descriptor | (1 << 7) | (1 << 10)),
                        T.uint32(descriptor),
                    ),
                    *decoded.operand("disable_output_lane"),
                    T.ptx.pred(T.uint32(0)),
                    pred=predicate,
                )
            )
        if isinstance(node, tirx.IfThenElse):
            mma_sites = []
            structural_walk(
                node.then_case,
                lambda child: mma_sites.append(child)
                if type(child).__name__ == "Call"
                and str(child.op.name).startswith("tirx.ptx.tcgen05_mma")
                else None,
            )
            if not mma_sites:
                return node

            def select_lane(expression):
                if type(expression).__name__ == "EQ" and str(expression.a) == "lane":
                    return (
                        tirx.LE(expression.a, T.int32(1))
                        if multiple_issuers
                        else tirx.EQ(expression.a, T.int32(7))
                    )
                return expression

            condition = structural_map(node.condition, select_lane)
            return tirx.IfThenElse(condition, node.then_case, node.else_case)
        return node

    kernel = kernel.with_body(structural_map(kernel.body, replace))
    inputs["metadata"][0, 0, :2] = (bf16, enabled)
    if input_descriptor:
        inputs["metadata"][0, 0, 0] = (
            descriptor_bits | (((1 << 7) | (1 << 10)) if bf16 else 0)
        ) ^ descriptor_xor
        if multiple_issuers:
            inputs["metadata"][0, 1, 0] = descriptor_bits | (1 << 7) | (1 << 10)
    if bf16:
        for name in ("a", "b"):
            values = inputs[name].view(np.float16).astype(np.float32)
            inputs[name] = (values.view(np.uint32) >> 16).astype(np.uint16)
    if not enabled:
        expected = inputs["seed"].copy()
    return kernel, inputs, expected


def replay(kernel, inputs, expected, *, output="out"):
    outputs = run_checked(kernel, inputs).outputs
    np.testing.assert_array_equal(outputs[output], expected)
    return outputs


def test_tcgen_runtime_descriptor_dispatch():
    for input_descriptor in (False, True):
        for bf16, enabled in ((False, True), (True, True), (True, False)):
            replay(*dispatch_case(bf16=bf16, enabled=enabled, input_descriptor=input_descriptor))
        kernel, inputs, _ = dispatch_case(
            multiple_issuers=True, input_descriptor=input_descriptor
        )
        assert_rejected(kernel, inputs, "requires exactly one issuing lane")

    # The descriptor is not a whitelist: signs stay runtime fields, and
    # malformed values are ignored only when the instruction is disabled.
    kernel, inputs, _ = dispatch_case(input_descriptor=True, descriptor_xor=1 << 13)
    negated_a = -inputs["a"][0].view(np.float16).astype(np.float32)
    b = inputs["b"].view(np.float16).astype(np.float32)
    replay(kernel, inputs, (negated_a @ b.T).view(np.int32))
    replay(*dispatch_case(input_descriptor=True, enabled=False, descriptor_xor=1 << 6))
    replay(*dispatch_case(input_descriptor=True, tmem_a=True))
    for descriptor_xor, tmem_a, verdict, message in (
        (1 << 6, False, "error", "descriptor must encode"),
        (1 << 10, False, "error", "mixed"),
        (1 << 15, True, "error", "transpose_a"),
    ):
        kernel, inputs, _ = dispatch_case(
            input_descriptor=True, descriptor_xor=descriptor_xor, tmem_a=tmem_a
        )
        assert_rejected(kernel, inputs, message, verdict=verdict)


@pytest.mark.numsim_gpu
def test_tcgen_runtime_descriptor_dispatch_gpu_oracle(pytestconfig):
    require_numsim_gpu(pytestconfig)
    for input_descriptor in (False, True):
        for bf16, enabled in ((False, True), (True, True), (True, False)):
            kernel, inputs, expected = dispatch_case(
                bf16=bf16, enabled=enabled, input_descriptor=input_descriptor
            )
            gpu = run_gpu_primfunc(kernel, inputs, outputs=("out",), arch="sm_100a")
            np.testing.assert_array_equal(gpu["out"], expected)


def _descriptor_values(kernel, descriptor):
    """The instruction-descriptor words ``kernel`` gives the MMA ``idesc`` operand."""

    if type(descriptor).__name__ == "IntImm":
        return {int(descriptor.value)}
    values = set()

    def collect(node):
        if type(node).__name__ != "Call" or str(node.op.name) != INSTR_DESC:
            return
        destination, *fields = node.args
        if type(descriptor).__name__ != "TensorLoad" or not destination.args[0].source.same_as(
            descriptor.source
        ):
            return
        d_dtype, a_dtype, b_dtype, m, n, k, trans_a, trans_b, cta_group, *flags = (
            field.value for field in fields
        )
        neg_a, neg_b, sat_d, sparse = map(bool, flags)
        values.add(
            encode_dense_instr_descriptor_fields(
                d_dtype=d_dtype, a_dtype=a_dtype, b_dtype=b_dtype, m=m, n=n, k=k,
                trans_a=bool(trans_a), trans_b=bool(trans_b), cta_group=cta_group,
                neg_a=neg_a, neg_b=neg_b, sat_d=sat_d, sparse=sparse,
            )
        )

    structural_walk(kernel.body, ((Expr, Stmt), collect))
    return values


def with_input_descriptor(kernel):
    parameter = tirx.Var("input_descriptor", "uint32")
    values = set()

    def replace(node):
        if type(node).__name__ != "Call" or not str(node.op.name).startswith(
            "tirx.ptx.tcgen05_mma"
        ):
            return node
        descriptor = decode_ptx_call(node).scalar_operand("idesc")
        resolved = _descriptor_values(kernel, descriptor)
        assert resolved
        values.update(resolved)
        return type(node)(
            node.op, [parameter if arg.same_as(descriptor) else arg for arg in node.args],
            attrs=node.attrs, ty_args=node.ty_args, span=node.span, ret_ty=node.ty,
        )

    body = structural_map(kernel.body, replace)
    assert len(values) == 1
    return (
        tirx.PrimFunc([*kernel.params, parameter], body, kernel.ret_type, kernel.attrs),
        values.pop(),
    )


def input_codec_cases():
    from tests.numsim.microtests.cases.tcgen05_mma_forms import (
        e5m2_layout_f_reference, f16_destination_reference,
        make_raw_e4m3_e5m2_arguments, make_raw_e5m2_arguments,
        make_raw_f16_destination_arguments, raw_e5m2_e4m3_f16_d_ss_m128_layout_d,
        raw_e5m2_ss_m64_layout_f_valid_descriptor,
    )
    from tests.numsim.runtime.test_tcgen05_i8 import i8_case

    kernel, descriptor = with_input_descriptor(raw_e5m2_ss_m64_layout_f_valid_descriptor)
    for make_inputs, a_format, a_dtype in (
        (make_raw_e5m2_arguments, 1, "float8_e5m2"),
        (make_raw_e4m3_e5m2_arguments, 0, "float8_e4m3fn"),
    ):
        inputs = make_inputs()
        inputs["input_descriptor"] = (descriptor & ~(7 << 7)) | (a_format << 7)
        yield kernel, inputs, e5m2_layout_f_reference(
            inputs, a_dtype=a_dtype, b_dtype="float8_e5m2"
        )
    kernel, descriptor = with_input_descriptor(raw_e5m2_e4m3_f16_d_ss_m128_layout_d)
    inputs = make_raw_f16_destination_arguments()
    inputs["input_descriptor"] = descriptor
    yield kernel, inputs, f16_destination_reference(inputs)
    kernel, inputs, expected = i8_case(1, 128, False, False, 1, 0, True, arch="sm_100a")
    kernel, descriptor = with_input_descriptor(kernel)
    inputs["input_descriptor"] = descriptor
    yield kernel, inputs, expected


def test_tcgen_input_descriptor_codecs():
    for kernel, inputs, expected in input_codec_cases():
        replay(kernel, inputs, expected, output="output" if "output" in inputs else "out")
    # The current TVM sparse entry requires SM107 collector syntax; keep the
    # existing model's dynamic-descriptor control off the SM100 GPU path.
    from tests.numsim.runtime.test_tcgen05_sparse_b16 import sparse_float_case

    kernel, inputs, expected = sparse_float_case(1, 64, False, 1, 1, False, False, False)
    kernel, descriptor = with_input_descriptor(kernel)
    inputs["input_descriptor"] = descriptor
    replay(kernel, inputs, expected)


@pytest.mark.numsim_gpu
def test_tcgen_input_descriptor_gpu_oracle(pytestconfig):
    require_numsim_gpu(pytestconfig)
    for kernel, inputs, expected in input_codec_cases():
        output = "output" if "output" in inputs else "out"
        gpu = run_gpu_primfunc(kernel, inputs, outputs=(output,), arch="sm_100a")
        np.testing.assert_array_equal(gpu[output], expected)
