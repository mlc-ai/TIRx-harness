from __future__ import annotations

import tvm
import tvm_ffi
import numpy as np
from tvm.tirx.buffer import is_buffer_var

from ..errors import NumSimExecutionError, UnsupportedTIRxError
from .ptx_dialect import PTX_SCHEMA_BY_OP_NAME, decode_ptx_call


SCOPE = "local.param"
PARAMETER_BUFFERS_ATTRIBUTE = "numsim.by_value_parameters"


def parameter_buffer_names(function):
    if function.attrs is None:
        return ()
    return tuple(str(name) for name in function.attrs.get(PARAMETER_BUFFERS_ATTRIBUTE, ()))


def snapshot_parameter_inputs(spec, contract, inputs):
    if not spec.parameter_buffers:
        return inputs
    snapshots = dict(inputs)
    for slot in contract.slots:
        if (
            slot.local_name not in spec.parameter_buffers[slot.kernel_index]
            or slot.canonical_name not in inputs
        ):
            continue
        value = inputs[slot.canonical_name]
        if not isinstance(value, np.ndarray):
            raise NumSimExecutionError(
                f"by-value parameter {slot.canonical_name!r} requires a NumPy array"
            )
        buffer = next(
            buffer
            for buffer in spec.kernels[slot.kernel_index].buffers
            if buffer.parameter == slot.local_name
        )
        expected_shape = tuple(int(extent) for extent in buffer.shape)
        if value.shape != expected_shape:
            raise NumSimExecutionError(
                f"by-value parameter {slot.canonical_name!r} requires shape {expected_shape}, "
                f"got {value.shape}"
            )
        if not value.flags.c_contiguous:
            raise NumSimExecutionError(
                f"by-value parameter {slot.canonical_name!r} requires contiguous storage"
            )
        snapshots[slot.canonical_name] = value.copy(order="C")
    return snapshots


def normalize_parameter_buffers(function):
    parameters = {
        value for value in function.params if is_buffer_var(value) and value.scope() == SCOPE
    }
    if not parameters:
        return function
    replacements = {}
    for parameter in parameters:
        if (
            len(parameter.shape) != 1
            or not isinstance(parameter.shape[0], tvm.tirx.IntImm)
            or not 0 < int(parameter.shape[0]) <= 1024
        ):
            raise UnsupportedTIRxError("by-value parameters require a fixed flat extent in 1..1024")
        if str(parameter.dtype) not in {"int32", "int64", "uint32", "uint64", "float32", "float64"}:
            raise UnsupportedTIRxError("unsupported by-value parameter element type")
        if not isinstance(parameter.elem_offset, tvm.tirx.IntImm) or int(parameter.elem_offset):
            raise UnsupportedTIRxError("by-value parameters require zero element offset")
        if parameter.strides and (
            len(parameter.strides) != 1
            or not isinstance(parameter.strides[0], tvm.tirx.IntImm)
            or int(parameter.strides[0]) != 1
        ):
            raise UnsupportedTIRxError("by-value parameters require contiguous storage")
        replacement = tvm.tirx.decl_buffer(
            parameter.shape,
            str(parameter.dtype),
            name=parameter.name,
            scope="global",
            data_alignment=parameter.data_alignment,
            offset_factor=parameter.offset_factor,
            elem_offset=parameter.elem_offset,
        )
        if parameter.layout is not None and not tvm_ffi.structural_equal(
            parameter.layout, replacement.layout
        ):
            raise UnsupportedTIRxError("by-value parameters require a plain contiguous layout")
        replacements[parameter] = replacement

    def parameter_load(value):
        return isinstance(value, tvm.ir.TensorLoad) and value.source in parameters

    def check(node):
        if is_buffer_var(node) and node.scope() == SCOPE and node not in parameters:
            raise UnsupportedTIRxError("by-value buffers must be direct parameters without aliases")
        if isinstance(node, tvm.tirx.BufferStore) and node.buffer in parameters:
            raise UnsupportedTIRxError("by-value parameter buffers are read-only")
        if not isinstance(node, tvm.ir.Call):
            return
        name = getattr(node.op, "name", "")
        if name in PTX_SCHEMA_BY_OP_NAME:
            decoded = decode_ptx_call(node)
            for slot in PTX_SCHEMA_BY_OP_NAME[name].operands:
                if "w" in slot.rw and any(
                    parameter_load(value) for value in decoded.operand(slot.name)
                ):
                    raise UnsupportedTIRxError("by-value parameter buffers are read-only")
        for argument in node.args:
            if is_buffer_var(argument) and argument in parameters:
                raise UnsupportedTIRxError("by-value parameter addresses cannot escape")
            if name in {"tirx.address_of", "prim.address_of"} and parameter_load(argument):
                raise UnsupportedTIRxError("by-value parameter addresses cannot escape")
            if isinstance(argument.ty, tvm.ir.PointerType) and argument.ty.storage_scope == SCOPE:
                raise UnsupportedTIRxError("by-value parameter addresses cannot escape")

    nodes = []
    tvm_ffi.structural_walk(function.body, nodes.append)
    for node in nodes:
        check(node)
    normalized = tvm_ffi.structural_map(
        function, (tvm.tirx.Var, lambda variable: replacements.get(variable, variable))
    )
    return normalized.with_attr(
        PARAMETER_BUFFERS_ATTRIBUTE, sorted(str(parameter.name) for parameter in parameters)
    )
