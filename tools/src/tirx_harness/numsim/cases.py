"""Explicit CPU-side kernel case bindings used by NumSim callers."""

from __future__ import annotations

import math
import operator
import struct
from collections.abc import Callable
from dataclasses import dataclass, field
from numbers import Real
from typing import Any

import numpy as np

from .dtype_abi import dtype_itemsize


def _buffer_dtype_itemsize(dtype: str) -> int | None:
    if dtype == "float4_e2m1fn":
        return 1
    if dtype in {"tf32", "float32_ftz", "tf32_ftz"}:
        return 4
    return dtype_itemsize(dtype)


def _tensor_map_element_bits(dtype: str) -> int:
    return {"float4_e2m1fn": 4, "uint6": 6}.get(dtype, 8 * (_buffer_dtype_itemsize(dtype) or 0))


def _plain_index(value: Any, *, field: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{field} must be an integer, not bool")
    try:
        return int(operator.index(value))
    except TypeError as error:
        raise TypeError(f"{field} must be an integer") from error


_TENSOR_MAP_DESCRIPTOR_BYTES = 128
_TENSOR_MAP_MAGIC = 0xA7
_TENSOR_MAP_FORMAT_TAGS = {None: _TENSOR_MAP_MAGIC, 16: 0xA6, 32: 0xA5}
# Swizzle mode and atomicity are independent descriptor fields: replacing the
# former must not erase the latter. Atomicity occupies tag bits 3 and 6.
_TENSOR_MAP_SWIZZLES = {
    None: (0, 0), "32B": (1, 0), "64B": (2, 0), "128B": (3, 0), "96B": (4, 0),
    "128B_ATOM_32B": (3, 1), "128B_ATOM_32B_FLIP_8B": (3, 2), "128B_ATOM_64B": (3, 3),
}
_TENSOR_MAP_FLAG_MAGIC = 0x80
_TENSOR_MAP_DTYPE_CODES = {
    "float4_e2m1fn": 0,
    "bool": 1,
    "int8": 2,
    "uint8": 3,
    "float8_e4m3fn": 4,
    "float8_e8m0fnu": 5,
    "int16": 6,
    "uint16": 7,
    "float16": 8,
    "bfloat16": 9,
    "int32": 10,
    "uint32": 11,
    "float32": 12,
    "tf32": 13,
    "float64": 14,
    "int64": 15,
    "uint64": 16,
    "uint32x2": 17,
    "float32_ftz": 18,
    "tf32_ftz": 19,
    "uint6": 20,
}


class _TensorMapArray(np.ndarray):
    """Descriptor array retaining only the lifetime of its addressed base."""

    def __array_finalize__(self, source: np.ndarray | None) -> None:
        self._tensor_map_base = getattr(source, "_tensor_map_base", None)


@dataclass(frozen=True)
class Im2col:
    """Spatial filter-base bounds, in W/H/D order; upper corners are end-relative.

    For wide mode only W has bounds. ``TensorMap.box_shape`` is the resulting
    two-dimensional column shape: (channels per pixel, pixels per column).
    """

    lower_corner: tuple[int, ...]
    upper_corner: tuple[int, ...]
    wide: bool = False


def _tensor_map_payload_bytes(tag: int) -> int:
    return 80 if tag & 0x10 else 64


@dataclass(frozen=True)
class TensorMap:
    """Build NumSim's copyable 128-byte TensorMap descriptor image.

    ``global_strides`` contains exactly ``rank - 1`` descriptor byte strides;
    the contiguous innermost element stride is implicit.
    ``tma_dtype`` may select ``tf32``, ``float32_ftz`` or ``tf32_ftz`` on
    float32 storage, or ``uint6`` on packed uint8 storage. FTZ affects tensor
    reductions, not ordinary copies. U6 has direction-specific shared packing.
    """

    base: np.ndarray
    global_shape: tuple[int, ...]
    global_strides: tuple[int, ...]
    box_shape: tuple[int, ...]
    element_strides: tuple[int, ...]
    dtype: str | None = None
    tma_dtype: str | None = None
    fp4_shared_layout: str | None = None
    swizzle: str | None = None
    interleave: str | None = None
    fill_mode: str | None = None
    im2col: Im2col | None = None

    def numpy(self) -> np.ndarray:
        """Return a copyable ``uint8[128]`` descriptor image."""

        base = np.asarray(self.base)
        if base.dtype.hasobject:
            raise ValueError("TensorMap base cannot use object dtype")
        pointer = int(base.__array_interface__["data"][0])
        if pointer == 0 and base.size:
            raise ValueError("TensorMap base exposes a null data pointer")
        if pointer % 16:
            raise ValueError("TensorMap global address must be 16-byte aligned")

        def integers(values: Any, field: str) -> tuple[int, ...]:
            try:
                result = tuple(_plain_index(value, field=field) for value in values)
            except TypeError as error:
                raise TypeError(f"TensorMap.{field} must contain integers") from error
            return result

        global_shape = integers(self.global_shape, "global_shape")
        global_strides = integers(self.global_strides, "global_strides")
        box_shape = integers(self.box_shape, "box_shape")
        element_strides = integers(self.element_strides, "element_strides")
        rank = len(global_shape)
        swizzle = None if self.swizzle in {None, "none"} else self.swizzle
        if swizzle not in _TENSOR_MAP_SWIZZLES:
            raise ValueError(f"unsupported TensorMap swizzle {swizzle!r}")
        swizzle_code, atomicity = _TENSOR_MAP_SWIZZLES[swizzle]
        interleave_bytes = {None: None, "none": None, "16B": 16, "32B": 32}.get(self.interleave, -1)
        if interleave_bytes == -1:
            raise ValueError(f"unsupported TensorMap interleave {self.interleave!r}")
        if interleave_bytes is not None and self.swizzle == "96B":
            raise ValueError("96B swizzle does not support interleave")
        if rank == 0 or rank > 5:
            raise ValueError("TensorMap rank must be in 1..5")
        if len(global_strides) != rank - 1:
            raise ValueError("TensorMap.global_strides must contain exactly rank-1 strides")
        if (
            len(box_shape) != (2 if self.im2col is not None else rank)
            or len(element_strides) != rank
        ):
            raise ValueError("TensorMap global/box/element-stride ranks disagree")
        if any(value <= 0 or value > 2**32 for value in global_shape):
            raise ValueError("TensorMap global dimensions must be in 1..2^32")
        if self.im2col is None and any(value <= 0 or value > 256 for value in box_shape):
            raise ValueError("TensorMap box dimensions must be in 1..256")
        if self.im2col is not None:
            if not isinstance(self.im2col, Im2col) or not isinstance(self.im2col.wide, bool):
                raise TypeError("TensorMap.im2col requires Im2col with a boolean wide flag")
            if rank < 3 or not (1 <= box_shape[0] <= 256 and 1 <= box_shape[1] <= 1024):
                raise ValueError("im2col requires rank 3..5, 1..256 channels and 1..1024 pixels")
            spatial_rank = 1 if self.im2col.wide else rank - 2
            lower = integers(self.im2col.lower_corner, "im2col.lower_corner")
            upper = integers(self.im2col.upper_corner, "im2col.upper_corner")
            bits = 16 if self.im2col.wide else {3: 16, 4: 8, 5: 5}[rank]
            if len(lower) != spatial_rank or len(upper) != spatial_rank:
                raise ValueError("im2col corner rank disagrees with tensor/mode")
            if any(
                not -(1 << (bits - 1)) <= value < (1 << (bits - 1)) for value in (*lower, *upper)
            ):
                raise ValueError(f"im2col corners must fit signed {bits}-bit values")
            if any(
                lo >= global_shape[i + int(interleave_bytes is None)] + hi
                for i, (lo, hi) in enumerate(zip(lower, upper))
            ):
                raise ValueError("im2col bounding box must have positive extent")
            if self.im2col.wide and interleave_bytes is not None:
                raise ValueError("wide im2col does not support interleave")
            if self.im2col.wide and (swizzle_code not in {2, 3, 4} or atomicity in {2, 3}):
                raise ValueError("wide im2col requires 64B, 96B or 128B swizzle")
        if any(value <= 0 or value >= 2**40 or value % 16 for value in global_strides):
            raise ValueError(
                "TensorMap global strides must be non-zero 16-byte multiples below 2^40"
            )
        if not 0 <= element_strides[0] <= 8 or any(
            value <= 0 or value > 8 for value in element_strides[1:]
        ):
            raise ValueError("TensorMap element strides must be in 1..8; axis zero may be zero")
        if interleave_bytes is None:
            element_strides = (1, *element_strides[1:])
        elif rank < 3 or element_strides[0] == 0:
            raise ValueError(
                "interleaved TensorMap requires rank 3..5 and nonzero axis-zero stride"
            )

        fp4_layout = None if self.fp4_shared_layout in {None, "none"} else self.fp4_shared_layout
        dtype = (
            "float4_e2m1fn"
            if self.dtype is None and fp4_layout is not None
            else str(base.dtype)
            if self.dtype is None
            else self.dtype
        )
        if self.tma_dtype not in {
            None, "none", "tf32", "tfloat32", "float32_ftz", "tf32_ftz", "uint6",
        }:
            raise ValueError(f"unsupported TensorMap TMA dtype {self.tma_dtype!r}")
        if self.tma_dtype == "uint6":
            if dtype != "uint8":
                raise ValueError("TensorMap U6 encoding requires packed uint8 storage")
            dtype = "uint6"
        elif self.tma_dtype not in {None, "none"}:
            if dtype != "float32":
                raise ValueError("TensorMap TMA float encoding requires a float32 base")
            dtype = "tf32" if self.tma_dtype == "tfloat32" else self.tma_dtype
        try:
            dtype_code = _TENSOR_MAP_DTYPE_CODES[dtype]
        except KeyError as error:
            raise ValueError(f"unsupported TensorMap dtype {dtype!r}") from error
        element_bits = _tensor_map_element_bits(dtype)
        if element_bits not in {4, 6, 8, 16, 32, 64}:
            raise ValueError(f"unsupported TensorMap element width {element_bits}")
        fp4_code = {None: 0, "align8_packed": 1, "align16_padded": 2}.get(fp4_layout)
        if fp4_code is None:
            raise ValueError(f"unsupported TensorMap FP4 shared layout {fp4_layout!r}")
        if fp4_layout == "align16_padded" and interleave_bytes is not None:
            raise ValueError("padded FP4 interleave is not modeled")
        if element_bits == 4 and fp4_code == 0:
            raise ValueError("FP4 TensorMap requires a shared layout")
        if element_bits != 4 and fp4_code != 0:
            raise ValueError("non-FP4 TensorMap cannot use an FP4 shared layout")
        if element_bits == 4 and base.dtype != np.dtype(np.uint8):
            raise ValueError("FP4 TensorMap requires a packed uint8 base")
        if element_bits == 6:
            if base.dtype != np.dtype(np.uint8):
                raise ValueError("U6 TensorMap requires a packed uint8 base")
            if interleave_bytes is not None:
                raise ValueError("U6 TensorMap does not support interleave")
            if global_shape[0] % 128 or box_shape[0] != 128:
                raise ValueError(
                    "SM100 U6 TensorMap requires dimension zero in multiples of 128 and box zero 128"
                )
            if pointer % 32 or any(stride % 32 for stride in global_strides):
                raise ValueError("SM100 U6 TensorMap address and strides must be 32-byte aligned")
            if swizzle_code not in {0, 3} or atomicity == 2:
                raise ValueError("SM100 U6 TensorMap supports only none or 128B swizzle")
        if fp4_layout == "align16_padded" and atomicity == 2:
            raise ValueError("padded FP4 TensorMap does not support 8B flip")
        if fp4_layout == "align8_packed" and global_shape[0] % 2:
            raise ValueError("align8 packed FP4 dimension zero must be even")
        if fp4_layout == "align16_padded" and (
            global_shape[0] % 128 or box_shape[0] != 128 or any(s % 32 for s in global_strides)
        ):
            raise ValueError("align16 padded FP4 metadata violates packed layout requirements")

        transfer_bits = interleave_bytes * 8 if interleave_bytes is not None else element_bits
        required_bytes = (global_shape[0] * transfer_bits + 7) // 8
        for stride, dimension in zip(global_strides, global_shape[1:], strict=True):
            required_bytes += (dimension - 1) * stride
        if base.nbytes < required_bytes:
            raise ValueError(f"TensorMap requires {required_bytes} base bytes, got {base.nbytes}")
        if (box_shape[0] * transfer_bits + 7) // 8 % 16:
            raise ValueError("TensorMap inner box transfer must be a multiple of 16 bytes")

        if interleave_bytes == 32 and (
            swizzle != "32B" or pointer % 32 or any(stride % 32 for stride in global_strides)
        ):
            raise ValueError("32B interleave requires 32B swizzle, address and strides")
        fill_mode = None if self.fill_mode in {None, "none", "zero"} else self.fill_mode
        if fill_mode not in {None, "nan"}:
            raise ValueError(f"unsupported TensorMap fill mode {fill_mode!r}")

        image = bytearray(_TENSOR_MAP_DESCRIPTOR_BYTES)
        image[0:8] = pointer.to_bytes(8, "little")
        image[8:16] = (0).to_bytes(8, "little")
        for axis in range(5):
            dimension = global_shape[axis] if axis < rank else 1
            encoded = 0 if dimension == 2**32 else dimension
            image[16 + axis * 4 : 20 + axis * 4] = encoded.to_bytes(4, "little")
        encoded_strides = [*global_strides, *([0] * (4 - len(global_strides)))]
        for pair in range(2):
            packed = (encoded_strides[pair * 2] >> 4) | ((encoded_strides[pair * 2 + 1] >> 4) << 36)
            image[36 + pair * 9 : 45 + pair * 9] = packed.to_bytes(9, "little")
        for axis in range(5):
            image[54 + axis] = ((box_shape[axis] if axis < len(box_shape) else 1) - 1) & 255
        image[59] = rank | (dtype_code << 3)
        image[60] = (
            fp4_code
            | ((swizzle_code & 3) << 2)
            | ((swizzle_code & 4) << 4)
            | (int(fill_mode == "nan") << 4)
            | (1 << 5)
            | _TENSOR_MAP_FLAG_MAGIC
        )
        encoded_element_strides = sum(
            (stride - 1) << (axis * 3) for axis, stride in enumerate(element_strides)
        )
        image[61:63] = struct.pack("<H", encoded_element_strides)
        image[63] = (
            _TENSOR_MAP_FORMAT_TAGS[interleave_bytes]
            | ((atomicity & 1) << 3)
            | ((atomicity & 2) << 5)
        )
        if self.im2col is not None:
            # Bit 4 extends the header; ordinary tiled maps keep their
            # 64-byte payload and reserved tail, even during replacement.
            image[63] |= 0x10
            struct.pack_into(
                "<3h3h",
                image,
                64,
                *lower,
                *([0] * (3 - len(lower))),
                *upper,
                *([0] * (3 - len(upper))),
            )
            image[76] = ((box_shape[1] - 1) >> 8) | (int(self.im2col.wide) << 2)
        result = np.frombuffer(image, dtype=np.uint8).copy().view(_TensorMapArray)
        result._tensor_map_base = base
        return result


def _descriptor_storage(*, storage: np.ndarray, slots: dict[int, np.ndarray]) -> np.ndarray:
    """Assemble TensorMap descriptors through ordinary NumPy byte copies."""

    result = np.asarray(storage)
    if not result.flags.c_contiguous:
        raise ValueError("TensorMap descriptor storage must be contiguous")
    bytes_view = result.view(np.uint8).reshape(-1)
    for offset, descriptor in slots.items():
        offset = _plain_index(offset, field="TensorMap descriptor byte offset")
        image = np.asarray(descriptor)
        if image.dtype != np.dtype(np.uint8) or image.shape != (128,):
            raise ValueError("TensorMap descriptor images must be uint8[128] arrays")
        if offset < 0 or offset % 128 or offset + 128 > bytes_view.size:
            raise ValueError("TensorMap descriptor byte offset is outside storage")
        bytes_view[offset : offset + 128] = image
    return result


@dataclass(frozen=True)
class ComparisonRegion:
    actual: tuple[int | slice, ...]
    expected: tuple[int | slice, ...] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "actual", self._normalize(self.actual, field="actual"))
        if self.expected is not None:
            object.__setattr__(self, "expected", self._normalize(self.expected, field="expected"))

    @staticmethod
    def _normalize(values: Any, *, field: str) -> tuple[int | slice, ...]:
        if not isinstance(values, tuple):
            raise TypeError(f"ComparisonRegion.{field} must be a tuple")
        result: list[int | slice] = []
        for value in values:
            if isinstance(value, slice):
                components: list[int | None] = []
                for component_name, component in (
                    ("start", value.start),
                    ("stop", value.stop),
                    ("step", value.step),
                ):
                    if component is None:
                        components.append(None)
                        continue
                    components.append(
                        _plain_index(
                            component, field=f"ComparisonRegion.{field} slice {component_name}"
                        )
                    )
                if components[2] == 0:
                    raise ValueError(f"ComparisonRegion.{field} slice step must not be zero")
                result.append(slice(*components))
            else:
                result.append(_plain_index(value, field=f"ComparisonRegion.{field} index"))
        return tuple(result)


@dataclass(frozen=True)
class ComparisonSpec:
    rtol: float = 1e-5
    atol: float = 1e-8
    equal_nan: bool = False
    actual_encoding: str | None = None
    regions: tuple[ComparisonRegion, ...] = ()

    def __post_init__(self) -> None:
        for field_name in ("rtol", "atol"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, Real):
                raise TypeError(f"ComparisonSpec.{field_name} must be a finite non-negative number")
            normalized = float(value)
            if not math.isfinite(normalized) or normalized < 0:
                raise ValueError(
                    f"ComparisonSpec.{field_name} must be a finite non-negative number"
                )
            object.__setattr__(self, field_name, normalized)
        if type(self.equal_nan) is not bool:
            raise TypeError("ComparisonSpec.equal_nan must be a bool")
        if self.actual_encoding not in {None, "bfloat16"}:
            raise ValueError(f"unsupported NumSim comparison encoding {self.actual_encoding!r}")
        if not isinstance(self.regions, tuple) or any(
            not isinstance(region, ComparisonRegion) for region in self.regions
        ):
            raise TypeError("ComparisonSpec.regions must be a tuple of ComparisonRegion values")


@dataclass(frozen=True)
class ExecutionAssumptions:
    """Optional external launch facts accepted by the execution API.

    External grid dependencies are satisfied automatically at an isolated
    NumSim launch boundary; explicit phase entries remain accepted for
    compatibility.
    """

    external_grid_dependencies_satisfied: tuple[int, ...] | list[int] = ()

    def __post_init__(self) -> None:
        values = self.external_grid_dependencies_satisfied
        if not isinstance(values, tuple | list):
            raise TypeError(
                "ExecutionAssumptions.external_grid_dependencies_satisfied must be a tuple or list"
            )
        normalized: list[int] = []
        for value in values:
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError("external grid dependency phase indices must be integers")
            if value < 0:
                raise ValueError("external grid dependency phase indices must be non-negative")
            normalized.append(value)
        if len(normalized) != len(set(normalized)):
            raise ValueError("external grid dependency phase indices contain duplicates")
        object.__setattr__(self, "external_grid_dependencies_satisfied", tuple(sorted(normalized)))


@dataclass
class NumSimCase:
    kernel: Any
    args: dict[str, np.ndarray | Any]
    outputs: tuple[str, ...] | dict[str, str] | None
    reference: Callable[[], dict[str, Any]]
    comparisons: dict[str, ComparisonSpec] = field(default_factory=dict)
    subset: Any | None = None
    assumptions: ExecutionAssumptions | None = None
