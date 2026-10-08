"""Matrix descriptors resolve from the selected bits, independent of their origin."""

import numpy as np
import pytest
from tvm import tirx
from tvm.ir import Call
from tvm.script import tirx as T
from tvm_ffi import structural_map

from tirx_harness import numsim, racecheck, synccheck
from tests.numsim.microtests.cases.tcgen05_lifecycle_ldst import tcgen05_cp_warpx4


@T.prim_func
def _publish_shared():
    T.ptx.fence.proxy.async_.shared__cta()


def descriptor_choice_case(form):
    kernel = tcgen05_cp_warpx4
    source = next(parameter for parameter in kernel.params if tirx.is_buffer_var(parameter))
    condition = source[0, 0] != tirx.const(0, "uint32")

    def replace(call):
        if str(call.op.name) != "tirx.ptx.tcgen05_cp":
            return call
        descriptor = call.args[1]
        reserved = tirx.const(1 << 14, "uint64")
        if form == "select":
            value = tirx.Select(condition, descriptor, tirx.bitwise_or(descriptor, reserved))
        else:
            low_word = tirx.Select(condition, tirx.const(0, "uint32"), tirx.const(1 << 14, "uint32"))
            value = tirx.bitwise_or(descriptor, tirx.Cast("uint64", low_word))
        return Call(
            call.op, [call.args[0], value, *call.args[2:]],
            attrs=call.attrs, ty_args=call.ty_args, span=call.span, ret_ty=call.ty,
        )

    def publish(stmt):
        if str(stmt.value.op.name) == "tirx.ptx.fence_mbarrier_init":
            # This fixture adds checker coverage to the GPU-only CP probe.
            return tirx.SeqStmt([_publish_shared.body, stmt])
        return stmt

    body = structural_map(kernel.body, (Call, replace))
    return kernel.with_body(structural_map(body, (tirx.Evaluate, publish)))


@pytest.mark.parametrize("form", ["select", "compose"])
@pytest.mark.parametrize("valid", [True, False])
def test_shared_descriptor_choices_validate_the_consumed_bits(tmp_path, form, valid):
    kernel = descriptor_choice_case(form)
    source = np.arange(128, dtype=np.uint32).reshape(32, 4) + 1
    source[0, 0] = int(valid)
    inputs = {"source": source, "output": np.full((4, 32, 4), 0xDEADBEEF, dtype=np.uint32)}
    for checker in (synccheck, racecheck):
        report = checker(kernel, inputs)
        if valid:
            report.require_clean()
        else:
            assert report.verdict == "error", report.to_dict()
            assert "reserved" in str(report.to_dict())
    module = numsim.transpile(kernel, cache_dir=tmp_path)
    if valid:
        result = numsim.Engine().run(module, inputs)
        np.testing.assert_array_equal(result.outputs["output"], np.tile(source, (4, 1, 1)))
    else:
        with pytest.raises(numsim.NumSimExecutionError, match="reserved"):
            numsim.Engine().run(module, inputs)
