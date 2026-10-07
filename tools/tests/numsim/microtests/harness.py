from __future__ import annotations

from collections.abc import Mapping, Sequence
import ctypes
from dataclasses import dataclass, replace
import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from tirx_harness import numsim


NUMSIM_GPU_MARK = pytest.mark.numsim_gpu


@dataclass(frozen=True)
class PairedRunResult:
    gpu_outputs: Mapping[str, np.ndarray]
    numsim_outputs: Mapping[str, np.ndarray]


@dataclass(frozen=True)
class PairedBuffer:
    """A physical host carrier with the logical dtype expected by the PrimFunc."""

    array: np.ndarray
    logical_dtype: str


@dataclass(frozen=True)
class PairedTensorMap:
    """One TensorMap definition shared by the GPU and NumSim executions."""

    array: np.ndarray
    global_shape: tuple[int, ...]
    global_strides: tuple[int, ...]
    box_shape: tuple[int, ...]
    element_strides: tuple[int, ...]
    logical_dtype: str | None = None
    tma_dtype: str | None = None
    fp4_shared_layout: str | None = None
    swizzle: str | None = None
    interleave: str | None = None
    fill_mode: str | None = None
    l2_promotion: str | None = None
    im2col: numsim.Im2col | None = None
    inactive_swizzle_atomicity: int = 0

    def __post_init__(self):
        if self.inactive_swizzle_atomicity not in range(4) or (
            self.inactive_swizzle_atomicity and self.swizzle not in {None, "none"}
        ):
            raise ValueError("inactive atomicity requires no swizzle and a PTX value in 0..3")


@dataclass(frozen=True)
class _GpuTensorMap:
    base: Any
    descriptor_storage: Any
    descriptor: Any


def require_numsim_gpu(
    pytestconfig: pytest.Config, *, min_nvrtc: tuple[int, int] | None = None
) -> None:
    """Skip when GPU microtests are disabled or the required hardware is unavailable."""

    enabled = bool(pytestconfig.getoption("run_numsim_gpu", default=True))
    if not enabled:
        pytest.skip("live NumSim/GPU microtests are disabled by --no-run-numsim-gpu")

    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("live NumSim/GPU microtests require a CUDA device")
    worker = os.environ.get("PYTEST_XDIST_WORKER", "gw0")
    if worker.startswith("gw") and worker[2:].isdigit():
        torch.cuda.set_device(int(worker[2:]) % torch.cuda.device_count())
    major, _minor = torch.cuda.get_device_capability()
    if major < 10:
        pytest.skip("live NumSim/GPU microtests require an SM100-or-newer GPU")
    if min_nvrtc is not None:
        from cuda.bindings import nvrtc

        status, major, minor = nvrtc.nvrtcVersion()
        assert status == nvrtc.nvrtcResult.NVRTC_SUCCESS
        if (major, minor) < min_nvrtc:
            pytest.skip(f"This instruction requires NVRTC {min_nvrtc[0]}.{min_nvrtc[1]} or newer")


def _clone_host_arguments(arguments: Mapping[str, Any]) -> dict[str, Any]:
    cloned: dict[str, Any] = {}
    for name, value in arguments.items():
        if isinstance(value, PairedBuffer):
            cloned[name] = PairedBuffer(
                np.array(value.array, copy=True, order="C"),
                value.logical_dtype,
            )
        elif isinstance(value, PairedTensorMap):
            cloned[name] = replace(
                value,
                array=np.array(value.array, copy=True, order="C"),
            )
        elif isinstance(value, np.ndarray):
            cloned[name] = np.array(value, copy=True, order="C")
        else:
            cloned[name] = value
    return cloned


def _torch_buffer(value: PairedBuffer):
    import torch

    storage = torch.from_numpy(np.ascontiguousarray(value.array))
    logical_dtype = {
        "bfloat16": torch.bfloat16,
        "float8_e4m3fn": torch.float8_e4m3fn,
        "float8_e8m0fnu": torch.float8_e8m0fnu,
    }.get(value.logical_dtype)
    if logical_dtype is None:
        raise TypeError(f"paired GPU buffer has unsupported logical dtype {value.logical_dtype!r}")
    return storage.view(logical_dtype)


def _logical_dtype_lanes(logical_dtype: str) -> int:
    import tvm

    return int(tvm.DataType(logical_dtype).lanes)


def _vector_base_dtype(logical_dtype: str) -> np.dtype:
    import tvm

    data_type = tvm.DataType(logical_dtype)
    if data_type.lanes <= 1:
        raise TypeError(f"logical dtype {logical_dtype!r} is not a vector dtype")
    suffix = f"x{data_type.lanes}"
    if not logical_dtype.endswith(suffix):
        raise TypeError(f"cannot derive vector element dtype from {logical_dtype!r}")
    base_dtype = logical_dtype[: -len(suffix)]
    if base_dtype == "bfloat16":
        import ml_dtypes  # noqa: F401 - registers NumPy's bfloat16 dtype

    return np.dtype(base_dtype)


def _tvm_vector_buffer(value: PairedBuffer):
    import tvm

    lanes = _logical_dtype_lanes(value.logical_dtype)
    base_dtype = _vector_base_dtype(value.logical_dtype)
    logical = np.ascontiguousarray(value.array).view(base_dtype).reshape(*value.array.shape, lanes)
    device = tvm.runtime.empty(value.array.shape, value.logical_dtype, tvm.cuda())
    device.copyfrom(logical)
    return device


def _torch_tensor_map(
    value: PairedTensorMap, *, base: Any | None = None, by_value: bool = True,
) -> _GpuTensorMap:
    import torch
    import tvm

    logical_dtype = value.logical_dtype or value.array.dtype.name
    if base is None:
        if logical_dtype != value.array.dtype.name:
            base = _torch_buffer(
                PairedBuffer(
                    value.array,
                    logical_dtype=logical_dtype,
                )
            ).to("cuda")
        else:
            base = torch.from_numpy(np.ascontiguousarray(value.array)).to("cuda")

    enum = {
        "interleave": {None: 0, "none": 0, "16B": 1, "32B": 2},
        "swizzle": {
            None: 0,
            "none": 0,
            "32B": 1,
            "64B": 2,
            "128B": 3,
            "128B_ATOM_32B": 4,
            "128B_ATOM_32B_FLIP_8B": 5,
            "128B_ATOM_64B": 6,
        },
        "l2_promotion": {None: 0, "none": 0, "64B": 1, "128B": 2, "256B": 3},
        "fill_mode": {None: 0, "none": 0, "zero": 0, "nan": 1},
        "tma_dtype": {"float32_ftz": 10, "tf32": 11, "tfloat32": 11, "tf32_ftz": 12, "uint6": 15},
    }

    def enum_value(field: str, raw: str | None) -> int:
        try:
            return enum[field][raw]
        except KeyError as error:
            raise ValueError(f"unsupported TensorMap {field} {raw!r}") from error

    tma_dtype = (
        None if value.tma_dtype in {None, "none"} else enum_value("tma_dtype", value.tma_dtype)
    )
    if tma_dtype is not None:
        required_dtype = "uint8" if tma_dtype == 15 else "float32"
        if logical_dtype != required_dtype:
            raise ValueError(f"TensorMap TMA encoding requires a {required_dtype} base")
    if value.fp4_shared_layout is not None:
        if tma_dtype is not None:
            raise ValueError("TensorMap FP4 layout cannot use a TMA float encoding")
        tma_dtype = {"align8_packed": 13, "align16_padded": 14}[value.fp4_shared_layout]

    # CUtensorMap must be 64-byte aligned and occupies 128 bytes.
    descriptor_storage = (ctypes.c_uint8 * (128 + 63))()
    descriptor_address = (ctypes.addressof(descriptor_storage) + 63) & ~63
    descriptor = ctypes.c_void_p(descriptor_address)

    def finish():
        argument = descriptor
        if value.inactive_swizzle_atomicity or not by_value:
            opaque = np.ctypeslib.as_array((ctypes.c_uint8 * 128).from_address(descriptor_address))
            device_image = torch.from_numpy(opaque.copy()).to("cuda")
        if value.inactive_swizzle_atomicity:
            from tvm.script import tirx as T

            # CUDA's encoder has no inactive-atomicity argument. Initialize
            # that independent field through PTX, never by guessing CUDA's
            # opaque descriptor bytes or modifying the kernel under test.
            source = f'''__device__ __forceinline__ void set_atomicity(void *p) {{
                asm volatile("tensormap.replace.tile.swizzle_atomicity.global.b1024.b32 [%0], {value.inactive_swizzle_atomicity};"
                             :: "l"(p) : "memory");
            }}'''
            initialize = tvm.script.from_source('''@T.prim_func
def initialize(image: T.Buffer((128,), "uint8")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    if lane == 0:
        T.cuda.func_call("set_atomicity", image.ptr_to([0]), source_code=source, return_type="void")
''', {"T": T, "source": source})
            major, minor = torch.cuda.get_device_capability()
            _compile_gpu(initialize, arch=f"sm_{major}{minor}a")(device_image)
            torch.cuda.synchronize()
            if by_value:
                opaque[:] = device_image.cpu().numpy()
        if not by_value:
            argument = device_image
        return _GpuTensorMap(base, descriptor_storage, argument)

    if value.im2col is not None:
        # TVM exposes only EncodeTiled; call CUDA's im2col encoder directly.
        dtype = {
            "uint8": 0,
            "uint16": 1,
            "uint32": 2,
            "int32": 3,
            "uint64": 4,
            "int64": 5,
            "float16": 6,
            "float32": 7,
            "float64": 8,
            "bfloat16": 9,
        }[logical_dtype]
        if tma_dtype is not None:
            dtype = tma_dtype
        rank = len(value.global_shape)
        wide = value.im2col.wide
        corners = (
            (ctypes.c_int(value.im2col.lower_corner[0]), ctypes.c_int(value.im2col.upper_corner[0]))
            if wide
            else (
                (ctypes.c_int * (rank - 2))(*value.im2col.lower_corner),
                (ctypes.c_int * (rank - 2))(*value.im2col.upper_corner),
            )
        )
        driver = ctypes.CDLL("libcuda.so.1")
        encode = driver.cuTensorMapEncodeIm2colWide if wide else driver.cuTensorMapEncodeIm2col
        status = encode(
            descriptor,
            ctypes.c_int(dtype),
            ctypes.c_uint(rank),
            ctypes.c_void_p(base.data_ptr()),
            (ctypes.c_uint64 * rank)(*value.global_shape),
            (ctypes.c_uint64 * (rank - 1))(*value.global_strides),
            *corners,
            ctypes.c_uint(value.box_shape[0]),
            ctypes.c_uint(value.box_shape[1]),
            (ctypes.c_uint32 * rank)(*value.element_strides),
            ctypes.c_int(enum_value("interleave", value.interleave)),
            # W encoding also works for the fixed-128 PTX instruction, which
            # ignores this descriptor's pixel count (verified on SM100).
            *((ctypes.c_int(0),) if wide else ()),
            ctypes.c_int(enum_value("swizzle", value.swizzle)),
            ctypes.c_int(enum_value("l2_promotion", value.l2_promotion)),
            ctypes.c_int(enum_value("fill_mode", value.fill_mode)),
        )
        if status:
            raise RuntimeError(f"CUDA im2col encoder failed with status {status}")
        return finish()
    if tma_dtype in {10, 12, 13, 14, 15}:
        # Use CUDA's explicit FTZ/sub-byte type, independently of the host carrier,
        # as for im2col and without changing the compiler.
        rank = len(value.global_shape)
        status = ctypes.CDLL("libcuda.so.1").cuTensorMapEncodeTiled(
            descriptor,
            ctypes.c_int(tma_dtype),
            ctypes.c_uint(rank),
            ctypes.c_void_p(base.data_ptr()),
            (ctypes.c_uint64 * rank)(*value.global_shape),
            (ctypes.c_uint64 * (rank - 1))(*value.global_strides),
            (ctypes.c_uint32 * rank)(*value.box_shape),
            (ctypes.c_uint32 * rank)(*value.element_strides),
            ctypes.c_int(enum_value("interleave", value.interleave)),
            ctypes.c_int(enum_value("swizzle", value.swizzle)),
            ctypes.c_int(enum_value("l2_promotion", value.l2_promotion)),
            ctypes.c_int(enum_value("fill_mode", value.fill_mode)),
        )
        if status:
            raise RuntimeError(f"CUDA tiled encoder failed with status {status}")
        return finish()
    encode = tvm.get_global_func("runtime.cuTensorMapEncodeTiled")
    encode_args = [
        descriptor,
        tvm.DataType(logical_dtype),
        len(value.global_shape),
        ctypes.c_void_p(base.data_ptr()),
        *value.global_shape,
        *value.global_strides,
        *value.box_shape,
        *value.element_strides,
        enum_value("interleave", value.interleave),
        enum_value("swizzle", value.swizzle),
        enum_value("l2_promotion", value.l2_promotion),
        enum_value("fill_mode", value.fill_mode),
    ]
    if tma_dtype is not None:
        encode_args.append(tma_dtype)
    encode(*encode_args)
    return finish()


def _numsim_tensor_map(value: PairedTensorMap) -> np.ndarray:
    image = numsim.TensorMap(
        base=value.array,
        global_shape=value.global_shape,
        global_strides=value.global_strides,
        box_shape=value.box_shape,
        element_strides=value.element_strides,
        dtype=None if value.fp4_shared_layout else value.logical_dtype,
        tma_dtype=value.tma_dtype,
        fp4_shared_layout=value.fp4_shared_layout,
        swizzle=value.swizzle,
        interleave=value.interleave,
        fill_mode=value.fill_mode,
        im2col=value.im2col,
    ).numpy()
    # NumSim's private image retains this field even with swizzle disabled.
    atom = value.inactive_swizzle_atomicity
    image[63] |= ((atom & 1) << 3) | ((atom & 2) << 5)
    return image


def _physical_gpu_output(device_value: Any, host_value: Any) -> np.ndarray:
    import torch

    if isinstance(host_value, PairedTensorMap):
        logical_dtype = host_value.logical_dtype or host_value.array.dtype.name
        if logical_dtype == host_value.array.dtype.name:
            return device_value.base.detach().cpu().numpy().copy()
        return _physical_gpu_output(
            device_value.base,
            PairedBuffer(
                host_value.array,
                logical_dtype,
            ),
        )

    if not isinstance(host_value, PairedBuffer):
        return device_value.detach().cpu().numpy().copy()

    if _logical_dtype_lanes(host_value.logical_dtype) > 1:
        logical = np.ascontiguousarray(device_value.numpy())
        return logical.view(host_value.array.dtype).reshape(host_value.array.shape).copy()

    storage_dtype = {
        np.dtype(np.uint8): torch.uint8,
        np.dtype(np.uint16): torch.uint16,
    }.get(host_value.array.dtype)
    if storage_dtype is None:
        raise TypeError(
            f"paired GPU output has unsupported physical carrier {host_value.array.dtype}"
        )
    storage = device_value.detach().view(storage_dtype)
    return storage.cpu().numpy().reshape(host_value.array.shape).copy()


def _parameter_names(prim_func: Any) -> tuple[str, ...]:
    return tuple(parameter.name for parameter in prim_func.params)


def _compile_gpu(prim_func: Any, *, arch: str) -> Any:
    import tvm

    target = tvm.target.Target({"kind": "cuda", "arch": arch})
    with target:
        return tvm.compile(
            tvm.IRModule({"main": prim_func}),
            target=target,
            tir_pipeline="tirx",
        )


def run_gpu_primfunc(
    prim_func: Any,
    arguments: Mapping[str, Any],
    *,
    outputs: Sequence[str],
    arch: str,
) -> dict[str, np.ndarray]:
    import torch
    from tvm.ir.type import PointerType
    from tvm.tirx import TensorMapType
    from tirx_harness.numsim.bindings import prepare_bindings

    parameter_names = _parameter_names(prim_func)
    # GPU execution must remain available for instructions NumSim cannot model.
    # Only the parameter type determines whether a descriptor is passed by value.
    tensor_map_names = {
        parameter.name for parameter in prim_func.params
        if isinstance(parameter.ty, PointerType)
        and isinstance(parameter.ty.element_type, TensorMapType)
    }
    missing = sorted(set(parameter_names) - set(arguments))
    extra = sorted(set(arguments) - set(parameter_names))
    if missing or extra:
        raise ValueError(f"GPU arguments do not match PrimFunc: missing={missing}, extra={extra}")

    device_arguments: dict[str, Any] = {}
    # Reuse the host binding owner's alias groups. Exact (pointer, size, dtype)
    # keys split a TensorMap's subview from its ordinary backing argument.
    bindings = prepare_bindings({
        name: value if isinstance(value, np.ndarray) else value.array
        for name, value in arguments.items()
        if isinstance(value, (np.ndarray, PairedBuffer, PairedTensorMap))
    }, expected_buffer_dtypes={
        name: value.logical_dtype
        for name, value in arguments.items()
        if isinstance(value, (PairedBuffer, PairedTensorMap)) and value.logical_dtype is not None
    })
    device_backings: dict[int, Any] = {}

    def device_array(name: str, value: PairedBuffer | PairedTensorMap | np.ndarray) -> Any:
        view = bindings.buffers[name]
        allocation = bindings.allocations[view.allocation]
        # Interior TensorMaps must keep their 16/32B alignment relative to the
        # backing. Standalone arrays/maps retain CUDA allocation alignment.
        has_interior_tensor_map = any(
            other.allocation == view.allocation and other.data_offset % 32
            and isinstance(arguments[key], PairedTensorMap)
            for key, other in bindings.buffers.items()
        )
        prefix = allocation.host_address % 32 if has_interior_tensor_map else 0
        if view.allocation not in device_backings:
            size = (prefix + allocation.byte_len + 31) // 32 * 32
            storage = torch.empty(size, dtype=torch.uint8, device="cuda")
            data = torch.from_numpy(np.frombuffer(allocation.data, np.uint8).copy())
            storage[prefix : prefix + allocation.byte_len].copy_(data)
            device_backings[view.allocation] = storage
        array = value if isinstance(value, np.ndarray) else value.array
        logical_dtype = None if isinstance(value, np.ndarray) else value.logical_dtype
        carrier = (
            _torch_buffer(PairedBuffer(array, logical_dtype))
            if logical_dtype and logical_dtype != array.dtype.name
            else torch.from_numpy(np.ascontiguousarray(array))
        )
        itemsize = carrier.element_size()
        if (prefix + view.data_offset) % itemsize or any(
            stride % itemsize for stride in view.byte_strides
        ):
            raise ValueError("GPU buffer offset and strides must be whole dtype elements")
        return device_backings[view.allocation].view(carrier.dtype).as_strided(
            array.shape, tuple(stride // itemsize for stride in view.byte_strides),
            storage_offset=(prefix + view.data_offset) // itemsize,
        )

    for name in parameter_names:
        value = arguments[name]
        if isinstance(value, PairedTensorMap):
            device_arguments[name] = _torch_tensor_map(
                value, base=device_array(name, value), by_value=name in tensor_map_names,
            )
        elif isinstance(value, PairedBuffer):
            if _logical_dtype_lanes(value.logical_dtype) > 1:
                device_arguments[name] = _tvm_vector_buffer(value)
            else:
                device_arguments[name] = device_array(name, value)
        elif isinstance(value, np.ndarray):
            device_arguments[name] = device_array(name, value)
        elif isinstance(value, np.generic):
            device_arguments[name] = value.item()
        else:
            device_arguments[name] = value

    executable = _compile_gpu(prim_func, arch=arch)
    executable(
        *(
            device_arguments[name].descriptor
            if isinstance(device_arguments[name], _GpuTensorMap)
            else device_arguments[name]
            for name in parameter_names
        )
    )
    torch.cuda.synchronize()
    return {name: _physical_gpu_output(device_arguments[name], arguments[name]) for name in outputs}


def run_paired_primfunc(
    prim_func: Any,
    arguments: Mapping[str, Any],
    *,
    outputs: Sequence[str],
    cache_dir: str | Path,
    arch: str = "sm_100a",
    max_ulp: int | Mapping[str, int] = 0,
) -> PairedRunResult:
    """Run one PrimFunc from identical host state on GPU and NumSim, then compare outputs."""

    output_names = tuple(outputs)
    if not output_names:
        raise ValueError("paired NumSim/GPU runs require at least one declared output")
    missing_outputs = sorted(set(output_names) - set(arguments))
    if missing_outputs:
        raise ValueError(f"declared outputs are absent from arguments: {missing_outputs}")

    initial = _clone_host_arguments(arguments)
    gpu_arguments = _clone_host_arguments(initial)
    numsim_arguments = _clone_host_arguments(initial)

    gpu_outputs = run_gpu_primfunc(
        prim_func,
        gpu_arguments,
        outputs=output_names,
        arch=arch,
    )
    module = numsim.transpile(prim_func, cache_dir=cache_dir)
    numsim_bindings = {
        name: (
            _numsim_tensor_map(value)
            if isinstance(value, PairedTensorMap)
            else value.array
            if isinstance(value, PairedBuffer)
            else value
        )
        for name, value in numsim_arguments.items()
    }
    result = numsim.Engine().run(module, numsim_bindings, outputs=output_names)
    numsim_outputs = {name: np.asarray(result.outputs[name]).copy() for name in output_names}

    def check_outputs() -> None:
        for name in output_names:
            if numsim_outputs[name].dtype != gpu_outputs[name].dtype:
                raise TypeError(
                    f"NumSim/GPU output dtype mismatch for {name!r}: "
                    f"{numsim_outputs[name].dtype} != {gpu_outputs[name].dtype}"
                )
            if numsim_outputs[name].shape != gpu_outputs[name].shape:
                raise AssertionError(
                    f"NumSim/GPU output shape mismatch for {name!r}: "
                    f"{numsim_outputs[name].shape} != {gpu_outputs[name].shape}"
                )
            output_max_ulp = max_ulp[name] if isinstance(max_ulp, Mapping) else max_ulp
            if output_max_ulp:
                np.testing.assert_array_max_ulp(
                    numsim_outputs[name],
                    gpu_outputs[name],
                    maxulp=output_max_ulp,
                )
            else:
                np.testing.assert_array_equal(
                    np.ascontiguousarray(numsim_outputs[name]).view(np.uint8),
                    np.ascontiguousarray(gpu_outputs[name]).view(np.uint8),
                    err_msg=f"NumSim/GPU output bytes mismatch for {name}",
                )

    check_outputs()

    return PairedRunResult(gpu_outputs=gpu_outputs, numsim_outputs=numsim_outputs)
