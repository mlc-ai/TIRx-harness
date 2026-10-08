"""Three-way GPU, NumSim, and independent-reference corpus validation."""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from tests.numsim.microtests.harness import (
    PairedBuffer,
    PairedTensorMap,
    run_gpu_primfunc,
)
from tirx_harness import numsim
from tirx_harness.numsim.bindings import (
    _decode_tensor_maps,
    _tensor_map_array_from_buffer,
    _tensor_map_physical_dtype,
)
from tirx_harness.numsim.host_abi import build_host_abi
from tirx_harness.numsim.transpiler.frontend import analyze


@dataclass(frozen=True)
class ThreeWayReport:
    gpu_outputs: dict[str, np.ndarray]
    numsim_outputs: dict[str, np.ndarray]
    gpu_vs_numsim: numsim.NumSimReport
    gpu_vs_reference: numsim.NumSimReport
    numsim_vs_reference: numsim.NumSimReport

    def require_ok(self) -> None:
        for relation, report in (
            ("GPU != NumSim", self.gpu_vs_numsim),
            ("GPU != reference", self.gpu_vs_reference),
            ("NumSim != reference", self.numsim_vs_reference),
        ):
            if report.ok:
                continue
            first = report.mismatches[0] if report.mismatches else None
            detail = "<missing mismatch detail>" if first is None else first.render()
            raise AssertionError(f"three-way corpus comparison failed: {relation}; {detail}")


def _host_tensor_map_base(value: np.ndarray) -> tuple[Any, np.ndarray] | None:
    descriptors = _decode_tensor_maps(value)
    if value.shape != (128,) or len(descriptors) != 1:
        return None
    descriptor = descriptors[0]
    raw = np.ctypeslib.as_array(
        (ctypes.c_uint8 * descriptor.required_byte_len).from_address(descriptor.address)
    )
    return descriptor, raw.view(_tensor_map_physical_dtype(descriptor.dtype))


def _clone_array(value: np.ndarray) -> np.ndarray:
    cloned = np.array(value, copy=True, order="K")
    # Preserve TensorMap's 16/32-byte alignment even when it points inside an
    # ordinary argument. NumPy's allocator only promises dtype alignment.
    if cloned.ctypes.data % 32 != value.ctypes.data % 32:
        owner = np.empty(cloned.nbytes + 31, dtype=np.uint8)
        aligned = np.ndarray(
            cloned.shape, dtype=cloned.dtype, buffer=owner,
            offset=(value.ctypes.data - owner.ctypes.data) % 32, strides=cloned.strides,
        )
        aligned[...] = cloned
        cloned = aligned
    return cloned


def _clone_tensor_map(
    value: np.ndarray,
    tensor_map: tuple[Any, np.ndarray],
    ordinary_arrays: list[tuple[np.ndarray, np.ndarray]],
    bases: list[np.ndarray],
) -> np.ndarray:
    descriptor, base = tensor_map
    cloned_address: int | None = None
    for original, cloned in ordinary_arrays:
        original_address = int(original.__array_interface__["data"][0])
        byte_offset = descriptor.address - original_address
        if 0 <= byte_offset and byte_offset + descriptor.required_byte_len <= original.nbytes:
            cloned_address = int(cloned.__array_interface__["data"][0]) + byte_offset
            break
    if cloned_address is None:
        cloned_base = _clone_array(base)
        bases.append(cloned_base)
        cloned_address = int(cloned_base.__array_interface__["data"][0])

    cloned_descriptor = np.array(value, copy=True)
    cloned_descriptor[:8] = np.frombuffer(
        cloned_address.to_bytes(8, "little"),
        dtype=np.uint8,
    )
    cloned_descriptor[8:16] = 0
    return cloned_descriptor


class _ClonedArguments(dict[str, Any]):
    tensor_map_bases: list[np.ndarray]


def _clone_arguments(arguments: dict[str, Any]) -> dict[str, Any]:
    bases: list[np.ndarray] = []
    values: dict[str, Any] = {}
    tensor_maps: dict[str, tuple[np.ndarray, tuple[Any, np.ndarray]]] = {}
    ordinary_arrays: list[tuple[np.ndarray, np.ndarray]] = []
    for name, value in arguments.items():
        if (
            isinstance(value, np.ndarray)
            and (tensor_map := _host_tensor_map_base(value)) is not None
        ):
            tensor_maps[name] = (value, tensor_map)
            continue
        cloned_value = _clone_array(value) if isinstance(value, np.ndarray) else value
        values[name] = cloned_value
        if isinstance(value, np.ndarray):
            ordinary_arrays.append((value, cloned_value))
    for name, (value, tensor_map) in tensor_maps.items():
        values[name] = _clone_tensor_map(value, tensor_map, ordinary_arrays, bases)

    cloned = _ClonedArguments({name: values[name] for name in arguments})
    cloned.tensor_map_bases = bases
    return cloned


def _paired_argument(value: Any) -> Any:
    if isinstance(value, np.ndarray) and (tensor_map := _host_tensor_map_base(value)) is not None:
        descriptor, base = tensor_map
        tma_dtype = (
            descriptor.dtype
            if descriptor.dtype in {"tf32", "float32_ftz", "tf32_ftz", "uint6"}
            else None
        )
        return PairedTensorMap(
            array=base,
            global_shape=descriptor.global_shape,
            global_strides=descriptor.global_strides,
            box_shape=descriptor.box_shape,
            element_strides=descriptor.element_strides,
            logical_dtype=base.dtype.name if tma_dtype or descriptor.fp4_shared_layout else descriptor.dtype,
            tma_dtype=tma_dtype,
            fp4_shared_layout=descriptor.fp4_shared_layout,
            swizzle=descriptor.swizzle,
            inactive_swizzle_atomicity=descriptor.inactive_swizzle_atomicity,
            fill_mode=descriptor.fill_mode,
            interleave=(
                f"{descriptor.interleave_bytes}B" if descriptor.interleave_bytes is not None else None
            ),
            im2col=descriptor.im2col,
        )
    return value


def _gpu_arguments(kernel: Any, arguments: dict[str, Any]) -> tuple[dict[str, Any], dict[str, str]]:
    gpu_arguments: dict[str, Any] = {}
    argument_to_gpu_name: dict[str, str] = {}
    consumed: set[str] = set()
    buffer_dtypes = build_host_abi(analyze(kernel)).buffer_dtypes
    for parameter in kernel.params:
        gpu_name = parameter.name
        candidates = [
            name for name in dict.fromkeys((parameter.name, gpu_name)) if name in arguments
        ]
        if len(candidates) != 1:
            raise ValueError(
                f"cannot map corpus argument for parameter {parameter.name!r}: candidates={candidates}"
            )
        argument_name = candidates[0]
        consumed.add(argument_name)
        argument_to_gpu_name[argument_name] = gpu_name
        paired = _paired_argument(arguments[argument_name])
        if isinstance(paired, np.ndarray):
            logical_dtype = buffer_dtypes.get(argument_name, paired.dtype.name)
            if logical_dtype != paired.dtype.name:
                paired = PairedBuffer(paired, logical_dtype)
        gpu_arguments[gpu_name] = paired
    extra = sorted(set(arguments) - consumed)
    if extra:
        raise ValueError(f"corpus arguments are absent from the PrimFunc parameters: {extra}")
    return gpu_arguments, argument_to_gpu_name


def _output_arguments(outputs: tuple[str, ...] | dict[str, str] | None) -> dict[str, str]:
    if outputs is None:
        raise ValueError("three-way corpus validation requires explicit outputs")
    if isinstance(outputs, dict):
        return dict(outputs)
    return {name: name for name in outputs}


def _gpu_output(argument: Any, output: np.ndarray) -> np.ndarray:
    """Return a GPU output through the same logical view as its TensorMap."""

    array = np.asarray(output)
    if not isinstance(argument, np.ndarray):
        return array.copy()
    descriptors = _decode_tensor_maps(argument)
    if argument.shape != (128,) or len(descriptors) != 1:
        return array.copy()
    descriptor = descriptors[0]
    raw = np.ascontiguousarray(array).view(np.uint8)
    return _tensor_map_array_from_buffer(
        raw,
        data_offset=0,
        global_shape=descriptor.physical_global_shape,
        global_strides=descriptor.global_strides,
        dtype=descriptor.dtype,
    ).copy()


def _peer_comparisons(
    comparisons: dict[str, numsim.ComparisonSpec],
) -> dict[str, numsim.ComparisonSpec]:
    result: dict[str, numsim.ComparisonSpec] = {}
    for name, spec in comparisons.items():
        regions = tuple(
            numsim.ComparisonRegion(actual=region.actual, expected=region.actual)
            for region in spec.regions
        )
        encoded_output = spec.actual_encoding is not None
        result[name] = numsim.ComparisonSpec(
            rtol=0.0 if encoded_output else spec.rtol,
            atol=0.0 if encoded_output else spec.atol,
            equal_nan=spec.equal_nan,
            actual_encoding=None,
            regions=regions,
        )
    return result


def run_three_way_case(
    case: numsim.NumSimCase,
    *,
    cache_dir: str | Path | None,
    arch: str = "sm_100a",
    engine: numsim.Engine | None = None,
) -> ThreeWayReport:
    """Run one complete kernel on GPU and NumSim and compare both to its reference."""

    expected = case.reference()
    if not expected:
        raise ValueError("three-way corpus validation requires a nonempty reference")

    gpu_case_arguments = _clone_arguments(case.args)
    numsim_arguments = _clone_arguments(case.args)
    gpu_arguments, argument_to_gpu_name = _gpu_arguments(case.kernel, gpu_case_arguments)
    output_arguments = _output_arguments(case.outputs)
    gpu_output_names = tuple(argument_to_gpu_name[name] for name in output_arguments.values())
    gpu_raw_outputs = run_gpu_primfunc(
        case.kernel,
        gpu_arguments,
        outputs=gpu_output_names,
        arch=arch,
    )
    gpu_outputs = {
        public_name: _gpu_output(
            gpu_case_arguments[argument_name],
            gpu_raw_outputs[argument_to_gpu_name[argument_name]],
        )
        for public_name, argument_name in output_arguments.items()
    }

    module = numsim.transpile(case.kernel, cache_dir=cache_dir)
    execution = (engine or numsim.Engine()).run(
        module,
        numsim_arguments,
        subset=case.subset,
        assumptions=case.assumptions,
        outputs=case.outputs,
    )
    numsim_outputs = {name: np.asarray(value).copy() for name, value in execution.outputs.items()}
    gpu_result = numsim.NumSimResult(outputs=gpu_outputs)
    peer_specs = _peer_comparisons(case.comparisons)
    return ThreeWayReport(
        gpu_outputs=gpu_outputs,
        numsim_outputs=numsim_outputs,
        gpu_vs_numsim=numsim.compare(execution, gpu_outputs, tolerances=peer_specs),
        gpu_vs_reference=numsim.compare(gpu_result, expected, tolerances=case.comparisons),
        numsim_vs_reference=numsim.compare(execution, expected, tolerances=case.comparisons),
    )
