"""The production frontend rejects unknown reflected statement kinds."""

import pytest
from tvm import tirx
from tvm_ffi import structural_map
from tvm_ffi.dataclasses import py_class

from tirx_harness.numsim.errors import NumSimBuildError
from tirx_harness.numsim.transpiler.frontend import analyze


def test_frontend_rejects_unknown_native_nodes_in_a_known_parent():
    @py_class("numsim.testing.FutureStatement")
    class FutureStatement(tirx.Stmt):
        body: tirx.Stmt

    known = tirx.PrimFunc([], tirx.SeqStmt([tirx.Evaluate(0), tirx.Evaluate(1)]))
    # Construct the extension through reflected fields: the current TVM
    # PrimFunc constructor's purity visitor cannot visit future node types.
    extended = structural_map(
        known,
        (
            tirx.Evaluate,
            lambda stmt: (
                FutureStatement(span=None, body=stmt) if int(stmt.value.value) == 1 else stmt
            ),
        ),
    )
    with pytest.raises(NumSimBuildError, match="node kind without a native binding"):
        analyze(extended)
    assert not analyze(known).kernels[0].unsupported
