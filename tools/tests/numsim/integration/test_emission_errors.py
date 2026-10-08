"""Analysis reports independent emission failures with source identities."""

import numpy as np
import pytest
import tvm
import tvm_ffi
from tvm.script import tirx as T

from tirx_harness import numsim
from tirx_harness.numsim.errors import UnsupportedTIRxError
from tirx_harness.numsim.transpiler.frontend import analyze, verify
from tirx_harness.numsim.transpiler import native_frontend
from tests.numsim.support.manifest import device_kernel


def _kernel(dtype):
    return tvm.script.from_source(
        f"""
@T.prim_func
def kernel(output: T.Buffer((1,), "float32")):
    T.device_entry()
    value: T.let = T.exp(T.{dtype}(1))
    output[0] = T.cast(value, "float32")
    T.evaluate(T.log(T.{dtype}(1)))
""",
        {"T": T},
    )


def test_analysis_collects_independent_failures_after_a_failed_binding():
    spec = analyze(_kernel("float64"))
    kernel = spec.kernels[0]
    entries = {entry.op_name: entry for entry in kernel.source_map if entry.op_name}
    assert len(kernel.unsupported) == 2
    for name in ("tirx.exp", "tirx.log"):
        source = entries[name]
        assert source.span is not None
        assert any(
            item.startswith(f"op#{source.op_id}:Call({name}(") for item in kernel.unsupported
        )
    assert all("unbound" not in item for item in kernel.unsupported)
    with pytest.raises(UnsupportedTIRxError):
        verify(spec)


def test_valid_signature_control_emits_and_executes(tmp_path):
    kernel = _kernel("float32")
    verify(analyze(kernel))
    module = numsim.transpile(kernel, cache_dir=tmp_path)
    result = numsim.Engine().run(module, {"output": np.zeros(1, dtype=np.float32)})
    np.testing.assert_allclose(result.outputs["output"], np.exp(np.float32(1)), rtol=1e-6)


_RECOVERY_CASES = [
    pytest.param(
        """value: T.let = T.exp(T.{dtype}(1)) + T.log(T.{dtype}(1))
    output[0] = T.cast(value, "float32")""",
        np.exp(np.float32(1)),
        id="sibling-expressions",
    ),
    pytest.param(
        """output[0] = T.cast(T.exp(T.log(T.{dtype}(1))), "float32")""",
        1,
        id="rejected-parent-call",
    ),
    pytest.param(
        """if T.exp(T.{dtype}(1)) > T.{dtype}(0):
        value: T.let = T.log(T.{dtype}(1))
        output[0] = T.cast(value, "float32")""",
        0,
        id="failed-condition-and-scoped-binding",
    ),
    pytest.param(
        """for i in T.serial(T.cast(T.exp(T.{dtype}(1)), "int32")):
        T.evaluate(T.exp(T.cast(i, "float32")))
        value: T.let = T.log(T.{dtype}(1))
        output[0] = T.cast(value, "float32") + T.cast(i, "float32")
        if i == 1:
            break""",
        1,
        id="failed-loop-bound-with-loop-variable-and-break",
    ),
    pytest.param(
        """while T.exp(T.{dtype}(1)) > T.{dtype}(0):
        T.evaluate(T.log(T.{dtype}(1)))
        output[0] = T.float32(3)
        break""",
        3,
        id="failed-while-condition",
    ),
]


def _recovery_kernel(body, dtype):
    return tvm.script.from_source(
        """
@T.prim_func
def kernel(output: T.Buffer((1,), "float32")):
    T.device_entry()
    """
        + body.format(dtype=dtype)
        + "\n",
        {"T": T},
    )


@pytest.mark.parametrize("body,expected", _RECOVERY_CASES)
def test_analysis_collects_unvisited_calls_after_an_emission_failure(body, expected):
    spec = analyze(_recovery_kernel(body, "float64"))
    kernel = spec.kernels[0]
    entries = {entry.op_id: entry for entry in kernel.source_map}
    assert len(kernel.unsupported) == 2, kernel.unsupported
    names = set()
    for message in kernel.unsupported:
        source = entries[int(message.split(":", 1)[0].removeprefix("op#"))]
        assert source.span is not None
        names.add(source.op_name)
        assert "expected" in message
    assert names == {"tirx.exp", "tirx.log"}
    with pytest.raises(UnsupportedTIRxError):
        verify(spec)


@pytest.mark.parametrize("body,expected", _RECOVERY_CASES)
def test_error_recovery_controls_keep_normal_emission_and_execution(body, expected, tmp_path):
    kernel = _recovery_kernel(body, "float32")
    verify(analyze(kernel))
    module = numsim.transpile(kernel, cache_dir=tmp_path)
    result = numsim.Engine().run(module, {"output": np.full(1, 7, dtype=np.float32)})
    np.testing.assert_allclose(result.outputs["output"], expected, rtol=1e-6)


@pytest.mark.parametrize("dtype", ["float64", "float32"])
def test_error_recovery_respects_expression_let_scope(dtype, tmp_path):
    output = tvm.tirx.decl_buffer((1,), "float32", name="output")
    variable = tvm.tirx.Var("scoped", "float32")
    bound = tvm.tirx.Let(variable, T.float32(1), T.exp(variable))
    expression = T.exp(T.cast(bound, dtype) + T.log(tvm.tirx.FloatImm(dtype, 1)))
    kernel = device_kernel(
        tvm.tirx.BufferStore(output, T.cast(expression, "float32"), [0]), (output,)
    )
    spec = analyze(kernel)
    errors = spec.kernels[0].unsupported
    # Let expressions are outside the public supported-node set, even though
    # the private expression emitter knows their binding scope.
    assert sum(message.endswith(":Let") for message in errors) == 1
    assert len(errors) == (3 if dtype == "float64" else 1), errors
    if dtype == "float64":
        assert any("Call(tirx.exp(float64)" in message for message in errors)
        assert any("Call(tirx.log(float64)" in message for message in errors)
    with pytest.raises(UnsupportedTIRxError):
        verify(spec)
    if dtype == "float32":
        # The equivalent supported expression is the successful control.
        expression = T.exp(T.exp(T.float32(1)) + T.log(T.float32(1)))
        kernel = device_kernel(tvm.tirx.BufferStore(output, expression, [0]), (output,))
        verify(analyze(kernel))
        module = numsim.transpile(kernel, cache_dir=tmp_path)
        result = numsim.Engine().run(module, {"output": np.zeros(1, dtype=np.float32)})
        np.testing.assert_allclose(
            result.outputs["output"], np.exp(np.exp(np.float32(1))), rtol=1e-6
        )


@pytest.mark.parametrize("threads", [64, 128])
def test_native_topology_returns_semantic_errors_as_data(threads):
    kernel = tvm.script.from_source(
        f"""
@T.prim_func
def kernel():
    T.device_entry()
    thread = T.thread_id_in_wg([{threads}])
    T.evaluate(thread)
""",
        {"T": T},
    )
    result = native_frontend._library()["numsim_launch_topology"](
        kernel, native_frontend.compile_schema(), tvm_ffi.convert({})
    )
    if threads == 64:
        assert result["value"] is None
        assert result["error"]["kind"] == "unsupported"
        assert "warpgroup thread extent must be 128" in result["error"]["message"]
        with pytest.raises(UnsupportedTIRxError, match="warpgroup thread extent must be 128"):
            native_frontend.launch_topology(kernel, {})
    else:
        assert result["error"] is None
        assert native_frontend.launch_topology(kernel, {})["warps_per_cta"] == 4


def test_foreign_exception_messages_are_not_parsed_as_frontend_errors(monkeypatch):
    error = ValueError('numsim-frontend:{"kind":"unsupported","message":"foreign error"}')

    def foreign_service():
        raise error

    monkeypatch.setattr(native_frontend, "_library", lambda: {"foreign": foreign_service})
    with pytest.raises(ValueError) as caught:
        native_frontend._call("foreign")
    assert caught.value is error
