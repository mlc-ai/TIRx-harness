"""Host binding normalization for the generated NumSim artifact ABI."""

from __future__ import annotations

import ctypes
import hashlib
import math
import struct
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import Any, Callable

import numpy as np

from .cases import (
    Im2col,
    _buffer_dtype_itemsize,
    _TENSOR_MAP_DESCRIPTOR_BYTES,
    _TENSOR_MAP_DTYPE_CODES,
    _tensor_map_element_bits,
    _TENSOR_MAP_FLAG_MAGIC,
    _TENSOR_MAP_FORMAT_TAGS,
    _TENSOR_MAP_SWIZZLES,
    _tensor_map_payload_bytes,
)
from .abi import NUMSIM_ABI_VERSION
from .errors import NumSimExecutionError

_PACKED_FLOAT4_DTYPE = "float4_e2m1fn"
_TENSOR_MAP_DTYPES = tuple(sorted(_TENSOR_MAP_DTYPE_CODES, key=_TENSOR_MAP_DTYPE_CODES.__getitem__))

_INTEGER_SCALAR_DTYPES = {
    "int8": (8, True),
    "int16": (16, True),
    "int32": (32, True),
    "int64": (64, True),
    "uint8": (8, False),
    "uint16": (16, False),
    "uint32": (32, False),
    "uint64": (64, False),
}
_FLOAT_SCALAR_DTYPES = {"float16", "bfloat16", "float32", "float64"}


@dataclass(frozen=True)
class PreparedAllocation:
    data: bytes | memoryview
    # An empty bitmap is the canonical all-valid sentinel. Allocation byte
    # length already distinguishes it from a genuinely empty allocation.
    validity: bytes
    label: str
    host_address: int

    def __post_init__(self) -> None:
        if self.validity and self.byte_len != len(self.validity):
            raise ValueError("allocation data and validity lengths differ")

    @property
    def byte_len(self) -> int:
        return self.data.nbytes if isinstance(self.data, memoryview) else len(self.data)

    @property
    def borrowed(self) -> bool:
        return isinstance(self.data, memoryview)

    def snapshot(self) -> bytes:
        return self.data.tobytes() if isinstance(self.data, memoryview) else self.data


@dataclass(frozen=True)
class PreparedBuffer:
    allocation: int
    data_offset: int
    dtype: str
    itemsize: int
    shape: tuple[int, ...]
    byte_strides: tuple[int, ...]

    @property
    def element_count(self) -> int:
        return math.prod(self.shape)

    @property
    def physical_byte_len(self) -> int:
        if self.dtype == _PACKED_FLOAT4_DTYPE:
            return (self.element_count + 1) // 2
        low, high = _byte_bounds(0, self.shape, self.byte_strides, self.itemsize)
        return high - low


@dataclass(frozen=True)
class PreparedTensorMapOutput:
    allocation: int
    data_offset: int
    global_shape: tuple[int, ...]
    global_strides: tuple[int, ...]
    dtype: str


@dataclass(frozen=True)
class PreparedScalar:
    value: int | float | bool
    dtype: str


@dataclass(frozen=True)
class _OriginalBuffer:
    array: np.ndarray
    pointer: int
    dtype: str
    shape: tuple[int, ...]
    byte_strides: tuple[int, ...]
    host_writeable: bool
    owner: Any
    root_array: np.ndarray
    root_pointer: int
    root_dtype: str
    root_shape: tuple[int, ...]
    root_byte_strides: tuple[int, ...]
    root_host_writeable: bool


@dataclass(frozen=True)
class _HostAllocation:
    implicit: bool
    address: int | None
    host_writeable: bool
    snapshot_is_immutable_owner: bool
    owners: tuple[Any, ...] = ()


@dataclass
class PreparedBindings:
    allocations: tuple[PreparedAllocation, ...]
    buffers: dict[str, PreparedBuffer]
    tensor_map_outputs: dict[str, PreparedTensorMapOutput]
    descriptor_allocations: frozenset[int]
    descriptor_storage_allocations: frozenset[int]
    scalars: dict[str, PreparedScalar]
    _originals: dict[str, _OriginalBuffer] = field(repr=False, compare=False)
    _host_allocations: tuple[_HostAllocation, ...] = field(repr=False, compare=False)

    def _write_through_allocations(self) -> frozenset[int]:
        return frozenset(
            index
            for index, (allocation, host) in enumerate(
                zip(self.allocations, self._host_allocations, strict=True)
            )
            if allocation.borrowed
            and host.implicit
            and host.host_writeable
            and index not in self.descriptor_storage_allocations
        )

    def freeze(self) -> PreparedBindings:
        """Own one exact byte snapshot for replay or non-mutating analysis."""

        if not any(allocation.borrowed for allocation in self.allocations):
            return self
        return replace(
            self,
            allocations=tuple(
                replace(allocation, data=allocation.snapshot())
                for allocation in self.allocations
            ),
        )

    def _require_stable_host_views(self) -> None:
        """Reject host metadata changes before using captured allocation addresses."""

        for name, original in self._originals.items():
            array = original.array
            owner, root_array = _array_owner(array)
            current = {
                "pointer": _array_pointer(array),
                "dtype": array.dtype.str,
                "shape": tuple(int(value) for value in array.shape),
                "byte_strides": tuple(int(value) for value in array.strides),
                "host_writeable": bool(array.flags.writeable),
                "owner": owner,
                "root_array": root_array,
                "root_pointer": _array_pointer(root_array),
                "root_dtype": root_array.dtype.str,
                "root_shape": tuple(int(value) for value in root_array.shape),
                "root_byte_strides": tuple(int(value) for value in root_array.strides),
                "root_host_writeable": bool(root_array.flags.writeable),
            }
            expected = {
                "pointer": original.pointer,
                "dtype": original.dtype,
                "shape": original.shape,
                "byte_strides": original.byte_strides,
                "host_writeable": original.host_writeable,
                "owner": original.owner,
                "root_array": original.root_array,
                "root_pointer": original.root_pointer,
                "root_dtype": original.root_dtype,
                "root_shape": original.root_shape,
                "root_byte_strides": original.root_byte_strides,
                "root_host_writeable": original.root_host_writeable,
            }
            changed = [
                field_name
                for field_name in expected
                if (
                    current[field_name] is not expected[field_name]
                    if field_name in {"owner", "root_array"}
                    else current[field_name] != expected[field_name]
                )
            ]
            if changed:
                raise NumSimExecutionError(
                    f"NumSim host view {name!r} changed after binding preparation: {changed}"
                )

    def identity_payload(
        self,
        *,
        _before_first_data_hash: Callable[[], None] | None = None,
    ) -> dict[str, Any]:
        """Return a process-independent identity for the frozen host snapshot."""

        canonical_data = [bytearray(allocation.snapshot()) for allocation in self.allocations]
        for name, buffer in self.buffers.items():
            original = self._originals[name].array
            if not original.flags.c_contiguous:
                continue
            byte_len = int(original.nbytes)
            raw = np.frombuffer(
                self.allocations[buffer.allocation].data,
                dtype=np.uint8,
                count=byte_len,
                offset=buffer.data_offset,
            )
            for descriptor in _decode_tensor_maps(raw):
                target = next(
                    (
                        (index, descriptor.address - allocation.host_address)
                        for index, allocation in enumerate(self.allocations)
                        if allocation.host_address <= descriptor.address
                        and descriptor.address + descriptor.required_byte_len
                        <= allocation.host_address + allocation.byte_len
                    ),
                    None,
                )
                if target is None:
                    raise NumSimExecutionError(
                        "TensorMap descriptor address is absent from prepared identity allocations"
                    )
                target_index, target_offset = target
                start = buffer.data_offset + descriptor.byte_offset
                canonical_data[buffer.allocation][start : start + 8] = target_index.to_bytes(
                    8, "little"
                )
                canonical_data[buffer.allocation][start + 8 : start + 16] = target_offset.to_bytes(
                    8, "little"
                )
                canonical_data[buffer.allocation][start + 60] &= ~(1 << 5)

        allocations: list[dict[str, Any]] = []
        before_first_data_hash = _before_first_data_hash
        for allocation, data in zip(self.allocations, canonical_data, strict=True):
            data_hash = hashlib.sha256()
            if before_first_data_hash is not None:
                before_first_data_hash()
                before_first_data_hash = None
            data_hash.update(data)
            allocations.append(
                {
                    "byte_len": allocation.byte_len,
                    "data_sha256": data_hash.hexdigest(),
                    "validity_sha256": (
                        hashlib.sha256(allocation.validity).hexdigest()
                        if allocation.validity
                        else "all-valid"
                    ),
                }
            )
        if before_first_data_hash is not None:
            before_first_data_hash()

        return {
            "numsim_abi_version": NUMSIM_ABI_VERSION,
            "allocations": allocations,
            "buffers": {
                name: {
                    "allocation": buffer.allocation,
                    "data_offset": buffer.data_offset,
                    "dtype": buffer.dtype,
                    "itemsize": buffer.itemsize,
                    "shape": list(buffer.shape),
                    "byte_strides": list(buffer.byte_strides),
                }
                for name, buffer in sorted(self.buffers.items())
            },
            "scalars": {
                name: {"dtype": scalar.dtype, "value": _scalar_identity_value(scalar.value)}
                for name, scalar in sorted(self.scalars.items())
            },
        }

    def with_allocation_state(self, states: Sequence[Any]) -> PreparedBindings:
        """Return bindings backed by an exact engine-produced allocation state."""

        if isinstance(states, (str, bytes, bytearray)) or not isinstance(states, Sequence):
            raise NumSimExecutionError("NumSim allocation state must be a sequence")
        if len(states) != len(self.allocations):
            raise NumSimExecutionError(
                "NumSim allocation state count changed: "
                f"expected {len(self.allocations)}, got {len(states)}"
            )
        allocations: list[PreparedAllocation] = []
        for index, (state, previous) in enumerate(zip(states, self.allocations, strict=True)):
            if not isinstance(state, dict):
                raise NumSimExecutionError(f"NumSim allocation state {index} must be a mapping")
            data = state.get("data")
            validity = state.get("validity")
            if not isinstance(data, bytes) or not isinstance(validity, bytes):
                raise NumSimExecutionError(
                    f"NumSim allocation state {index} data and validity must be bytes"
                )
            if len(data) != previous.byte_len or len(validity) != previous.byte_len:
                raise NumSimExecutionError(f"NumSim allocation state {index} changed byte length")
            if any(value not in (0, 1) for value in validity):
                raise NumSimExecutionError(
                    f"NumSim allocation state {index} has a non-binary validity byte"
                )
            allocations.append(
                PreparedAllocation(
                    data=data,
                    validity=validity,
                    label=previous.label,
                    host_address=previous.host_address,
                )
            )
        return replace(self, allocations=tuple(allocations))

    def logical_snapshot(self, name: str) -> np.ndarray:
        """Rebuild one logical array using the kernel-declared element dtype."""

        try:
            descriptor = self.buffers[name]
        except KeyError as error:
            raise NumSimExecutionError(f"unknown NumSim buffer {name!r}") from error
        allocation = self.allocations[descriptor.allocation]
        if descriptor.dtype == _PACKED_FLOAT4_DTYPE:
            packed = np.frombuffer(
                allocation.data,
                dtype=np.uint8,
                count=descriptor.physical_byte_len,
                offset=descriptor.data_offset,
            )
            codes = np.empty(descriptor.element_count, dtype=np.uint8)
            codes[0::2] = packed & np.uint8(0x0F)
            codes[1::2] = (packed[: descriptor.element_count // 2] >> np.uint8(4)) & np.uint8(0x0F)
            magnitudes = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32)
            values = magnitudes[codes & np.uint8(0x07)]
            values = np.where(codes & np.uint8(0x08), -values, values)
            return values.reshape(descriptor.shape)
        try:
            logical_dtype = np.dtype(descriptor.dtype)
        except TypeError:
            logical_dtype = None
        if logical_dtype is not None:
            return np.ndarray(
                shape=descriptor.shape,
                dtype=logical_dtype,
                buffer=allocation.data,
                offset=descriptor.data_offset,
                strides=descriptor.byte_strides,
            ).copy()
        if descriptor.dtype == "bfloat16":
            bits = np.ndarray(
                shape=descriptor.shape,
                dtype=np.uint16,
                buffer=allocation.data,
                offset=descriptor.data_offset,
                strides=descriptor.byte_strides,
            ).copy()
            return (bits.astype(np.uint32) << np.uint32(16)).view(np.float32)
        raise NumSimExecutionError(
            f"NumSim cannot materialize logical dtype {descriptor.dtype!r} "
            f"for buffer {name!r} without changing its values"
        )

    def to_payload(
        self, *, returned_names: set[str] | None = None, borrow_readonly: bool = False
    ) -> dict[str, Any]:
        """Serialize the bindings for one native call.

        With ``borrow_readonly`` nothing is written through: the borrowed host
        arrays are handed to the engine as read-only initial bytes (its
        allocations copy on write), so a non-mutating analysis neither copies
        the inputs nor touches the caller's arrays.
        """

        returned_names = (
            set(self.buffers) | set(self.tensor_map_outputs)
            if returned_names is None
            else returned_names
        )
        unknown = returned_names - self.buffers.keys() - self.tensor_map_outputs.keys()
        if unknown:
            raise NumSimExecutionError(
                f"NumSim requested unknown returned buffers: {sorted(unknown)}"
            )
        write_through_allocations = (
            frozenset() if borrow_readonly else self._write_through_allocations()
        )
        readonly_allocations = (
            frozenset(
                index
                for index, (allocation, host) in enumerate(
                    zip(self.allocations, self._host_allocations, strict=True)
                )
                if allocation.borrowed and host.owners
            )
            if borrow_readonly
            else frozenset()
        )
        host_backed = write_through_allocations | readonly_allocations
        return {
            "numsim_abi_version": NUMSIM_ABI_VERSION,
            "allocations": [
                {
                    "data": None if index in host_backed else allocation.snapshot(),
                    "byte_len": allocation.byte_len,
                    "validity": allocation.validity,
                    "label": allocation.label,
                    "host_address": allocation.host_address,
                    "write_through": index in write_through_allocations,
                    "host_readonly": index in readonly_allocations,
                    "host_buffer": allocation.data if index in host_backed else None,
                    "host_owner": host.owners if index in host_backed else None,
                }
                for index, (allocation, host) in enumerate(
                    zip(self.allocations, self._host_allocations, strict=True)
                )
            ],
            "output_allocations": sorted(
                {
                    *(
                        self.buffers[name].allocation
                        for name in returned_names
                        if name in self.buffers
                    ),
                    *(
                        self.tensor_map_outputs[name].allocation
                        for name in returned_names
                        if name in self.tensor_map_outputs
                    ),
                    *self.descriptor_allocations,
                }
            ),
            # The allocations the kernel may write: the returned buffers and the
            # bases of the returned TensorMaps. `output_allocations` also holds
            # every descriptor's base and storage so execution can hand them
            # back; an analysis that tracks written global memory would
            # otherwise shadow every read-only TensorMap source.
            "written_allocations": sorted(
                {
                    *(
                        self.buffers[name].allocation
                        for name in returned_names
                        if name in self.buffers
                    ),
                    *(
                        self.tensor_map_outputs[name].allocation
                        for name in returned_names
                        if name in self.tensor_map_outputs
                    ),
                }
            ),
            "buffers": {
                name: {
                    "allocation": buffer.allocation,
                    "data_offset": buffer.data_offset,
                    "dtype": buffer.dtype,
                    "itemsize": buffer.itemsize,
                    "shape": list(buffer.shape),
                    "byte_strides": list(buffer.byte_strides),
                }
                for name, buffer in sorted(self.buffers.items())
            },
            "scalars": {
                name: {"value": scalar.value, "dtype": scalar.dtype}
                for name, scalar in sorted(self.scalars.items())
            },
        }

    def restore_host_buffers(self) -> None:
        """Restore host views to the bytes captured during preparation."""

        self._require_stable_host_views()
        for allocation, host in zip(self.allocations, self._host_allocations):
            if not host.implicit:
                continue
            if allocation.borrowed:
                raise NumSimExecutionError(
                    "cannot restore a borrowed NumSim allocation; freeze bindings before replay"
                )
            assert host.address is not None
            if allocation.data:
                if not host.host_writeable:
                    if host.snapshot_is_immutable_owner:
                        continue
                    current = _snapshot_host_bytes(host.address, allocation.byte_len)
                    if current != allocation.data:
                        raise NumSimExecutionError(
                            "cannot restore a changed implicit allocation backed only by "
                            "read-only host views"
                        )
                    continue
                ctypes.memmove(host.address, allocation.data, allocation.byte_len)

        for name, descriptor in self.buffers.items():
            original = self._originals[name]
            if self._host_allocations[descriptor.allocation].implicit:
                continue
            allocation = self.allocations[descriptor.allocation]
            if descriptor.dtype == _PACKED_FLOAT4_DTYPE:
                packed = np.frombuffer(
                    allocation.data,
                    dtype=np.uint8,
                    count=descriptor.physical_byte_len,
                    offset=descriptor.data_offset,
                ).reshape(original.array.shape)
                if not original.array.flags.writeable:
                    if not np.array_equal(original.array, packed):
                        raise NumSimExecutionError(
                            f"cannot restore changed explicit read-only host view {name!r}"
                        )
                    continue
                original.array[...] = packed
                continue
            logical = np.ndarray(
                shape=descriptor.shape,
                dtype=original.array.dtype,
                buffer=allocation.data,
                offset=descriptor.data_offset,
                strides=descriptor.byte_strides,
            )
            if not original.array.flags.writeable:
                current = np.ascontiguousarray(original.array).tobytes(order="C")
                expected = np.ascontiguousarray(logical).tobytes(order="C")
                if current != expected:
                    raise NumSimExecutionError(
                        f"cannot restore changed explicit read-only host view {name!r}"
                    )
                continue
            original.array[...] = logical.reshape(original.array.shape)

    def apply_allocation_bytes(
        self,
        allocation_bytes: list[bytes | None] | tuple[bytes | None, ...],
        *,
        output_names: set[str] | None = None,
    ) -> dict[str, np.ndarray]:
        """Apply owned outputs and materialize write-through host outputs."""
        self._require_stable_host_views()
        if len(allocation_bytes) != len(self.allocations):
            raise NumSimExecutionError(
                "NumSim returned a different physical allocation count: "
                f"expected {len(self.allocations)}, got {len(allocation_bytes)}"
            )
        write_through_allocations = self._write_through_allocations()
        copied: list[bytes | memoryview | None] = []
        for index, (actual, expected) in enumerate(zip(allocation_bytes, self.allocations)):
            if actual is None:
                copied.append(
                    expected.data if index in write_through_allocations else None
                )
                continue
            value = bytes(actual)
            if len(value) != expected.byte_len:
                raise NumSimExecutionError(
                    f"NumSim allocation {index} changed byte length from "
                    f"{expected.byte_len} to {len(value)}"
                )
            copied.append(value)

        for index, (physical, host) in enumerate(
            zip(copied, self._host_allocations, strict=True)
        ):
            if physical is None or not host.implicit:
                continue
            if index in write_through_allocations:
                continue
            assert host.address is not None
            if physical and host.host_writeable:
                ctypes.memmove(host.address, physical, len(physical))

        outputs: dict[str, np.ndarray] = {}
        selected = output_names if output_names is not None else set(self.buffers)
        for name, descriptor in self.buffers.items():
            original = self._originals[name]
            physical = copied[descriptor.allocation]
            if physical is None:
                if name in selected:
                    raise NumSimExecutionError(
                        f"NumSim omitted requested allocation {descriptor.allocation} for {name!r}"
                    )
                continue
            if descriptor.dtype == _PACKED_FLOAT4_DTYPE:
                logical = (
                    np.frombuffer(
                        physical,
                        dtype=np.uint8,
                        count=descriptor.physical_byte_len,
                        offset=descriptor.data_offset,
                    )
                    .copy()
                    .reshape(original.array.shape)
                )
            else:
                logical = np.ndarray(
                    shape=descriptor.shape,
                    dtype=original.array.dtype,
                    buffer=physical,
                    offset=descriptor.data_offset,
                    strides=descriptor.byte_strides,
                ).copy()
            if not self._host_allocations[descriptor.allocation].implicit:
                target = original.array
                if target.size != logical.size:
                    raise NumSimExecutionError(
                        f"NumSim output {name!r} has {logical.size} elements but its host "
                        f"binding has {target.size}"
                    )
                target[...] = logical.reshape(target.shape)
            if name in selected:
                outputs[name] = logical
        for name, descriptor in self.tensor_map_outputs.items():
            if name not in selected:
                continue
            physical = copied[descriptor.allocation]
            if physical is None:
                raise NumSimExecutionError(
                    f"NumSim omitted TensorMap base allocation {descriptor.allocation} for {name!r}"
                )
            outputs[name] = _tensor_map_array_from_buffer(
                physical,
                data_offset=descriptor.data_offset,
                global_shape=descriptor.global_shape,
                global_strides=descriptor.global_strides,
                dtype=descriptor.dtype,
            ).copy()
        return outputs


@dataclass
class _PendingBuffer:
    name: str
    array: np.ndarray
    dtype: str
    itemsize: int
    shape: tuple[int, ...]
    byte_strides: tuple[int, ...]
    origin: int
    low: int
    high: int
    owner: Any
    root_array: np.ndarray
    backing_low: int
    backing_high: int
    host_low: int
    host_high: int
    borrowable: bool
    public: bool = True


def _scalar_identity_value(value: int | float | bool) -> dict[str, Any]:
    if isinstance(value, bool):
        return {"kind": "bool", "value": value}
    if isinstance(value, int):
        return {"kind": "integer", "decimal": str(value)}
    if isinstance(value, float):
        return {"kind": "float64-bits", "hex": struct.pack(">d", value).hex()}
    raise TypeError(f"unsupported prepared scalar identity value {type(value).__name__}")


def _f32_bits(value: float) -> int:
    return struct.unpack(">I", struct.pack(">f", value))[0]


def _f32_from_bits(bits: int) -> float:
    return struct.unpack(">f", struct.pack(">I", bits))[0]


def _round_shift_right_even(value: int, shift: int) -> int:
    if shift == 0:
        return value
    truncated = value >> shift
    remainder = value & ((1 << shift) - 1)
    halfway = 1 << (shift - 1)
    round_up = remainder > halfway or (remainder == halfway and truncated & 1 != 0)
    return truncated + int(round_up)


def _f32_to_fp16_bits(value: float) -> int:
    bits = _f32_bits(value)
    sign = (bits >> 16) & 0x8000
    exponent = (bits >> 23) & 0xFF
    fraction = bits & 0x007F_FFFF
    if exponent == 0xFF:
        if fraction == 0:
            return sign | 0x7C00
        return sign | 0x7C00 | (fraction >> 13) | 0x0200
    if exponent == 0:
        return sign
    half_exponent = exponent - 127 + 15
    if half_exponent >= 0x1F:
        return sign | 0x7C00
    if half_exponent <= 0:
        if half_exponent < -10:
            return sign
        significand = fraction | 0x0080_0000
        return sign | _round_shift_right_even(significand, 14 - half_exponent)
    rounded_fraction = _round_shift_right_even(fraction, 13)
    encoded = (half_exponent << 10) + rounded_fraction
    return sign | (0x7C00 if encoded >= 0x7C00 else encoded)


def _fp16_bits_to_f32(bits: int) -> float:
    sign = (bits & 0x8000) << 16
    exponent = (bits >> 10) & 0x1F
    fraction = bits & 0x03FF
    if exponent == 0 and fraction == 0:
        decoded = sign
    elif exponent == 0:
        leading_zeros = 10 - fraction.bit_length()
        normalized_fraction = (fraction << (leading_zeros + 1)) & 0x03FF
        decoded = sign | ((127 - 15 - leading_zeros) << 23) | (normalized_fraction << 13)
    elif exponent == 0x1F:
        decoded = sign | 0x7F80_0000 | (fraction << 13)
    else:
        decoded = sign | ((exponent + (127 - 15)) << 23) | (fraction << 13)
    return _f32_from_bits(decoded)


def _f32_to_bf16_bits(value: float) -> int:
    bits = _f32_bits(value)
    exponent = bits & 0x7F80_0000
    fraction = bits & 0x007F_FFFF
    if exponent == 0x7F80_0000 and fraction != 0:
        return (bits >> 16) | 0x0040
    tie = (bits >> 16) & 1
    return ((bits + 0x7FFF + tie) & 0xFFFF_FFFF) >> 16


def _normalize_float_scalar(value: float, dtype: str) -> float:
    if dtype == "float64":
        return value
    rounded_f32 = _f32_from_bits(_f32_bits(value))
    if dtype == "float32":
        return rounded_f32
    if dtype == "float16":
        return _fp16_bits_to_f32(_f32_to_fp16_bits(rounded_f32))
    if dtype == "bfloat16":
        return _f32_from_bits(_f32_to_bf16_bits(rounded_f32) << 16)
    raise NumSimExecutionError(f"unsupported NumSim scalar dtype {dtype!r}")


def _byte_bounds(
    origin: int, shape: tuple[int, ...], strides: tuple[int, ...], itemsize: int
) -> tuple[int, int]:
    if any(extent < 0 for extent in shape):
        raise NumSimExecutionError(f"buffer shape contains a negative extent: {shape}")
    if any(extent == 0 for extent in shape):
        return origin, origin
    low = origin
    high = origin
    for extent, stride in zip(shape, strides):
        delta = (extent - 1) * stride
        if delta < 0:
            low += delta
        else:
            high += delta
    return low, high + itemsize


def _array_owner(array: np.ndarray) -> tuple[Any, np.ndarray]:
    owner: Any = array
    root_array = array
    seen: set[int] = set()
    while id(owner) not in seen:
        seen.add(id(owner))
        if isinstance(owner, np.ndarray):
            root_array = owner
        base = getattr(owner, "base", None)
        if base is None and isinstance(owner, memoryview):
            base = owner.obj
        if base is None:
            break
        owner = base
    return owner, root_array


def _array_pointer(array: np.ndarray) -> int:
    interface = array.__array_interface__
    pointer = int(interface["data"][0])
    if pointer == 0 and array.size:
        raise NumSimExecutionError("NumPy binding exposes a null data pointer")
    return pointer


def _host_byte_view(address: int, byte_len: int) -> memoryview:
    if byte_len == 0:
        return memoryview(bytearray())
    raw = (ctypes.c_uint8 * byte_len).from_address(address)
    return memoryview(raw).cast("B")


def _snapshot_host_bytes(address: int, byte_len: int) -> bytes:
    return _host_byte_view(address, byte_len).tobytes()


def _dtype_name_and_size(
    name: str, array: np.ndarray, expected_dtype: str | None
) -> tuple[str, int]:
    dtype = str(array.dtype) if expected_dtype is None else expected_dtype
    if not isinstance(dtype, str):
        raise NumSimExecutionError(f"unsupported NumSim buffer dtype {dtype!r}")
    if dtype == _PACKED_FLOAT4_DTYPE:
        if array.dtype != np.dtype(np.uint8) or not array.flags.c_contiguous:
            raise NumSimExecutionError(
                f"NumSim packed float4 buffer {name!r} requires a contiguous uint8 array"
            )
        return dtype, 1
    itemsize = _buffer_dtype_itemsize(dtype)
    if itemsize is None:
        raise NumSimExecutionError(f"unsupported NumSim buffer dtype {dtype!r}")
    if itemsize != int(array.dtype.itemsize):
        raise NumSimExecutionError(
            f"declared dtype {dtype!r} has itemsize {itemsize}, but host array "
            f"dtype {array.dtype} has itemsize {array.dtype.itemsize}"
        )
    return dtype, itemsize


def _normalize_buffer(
    name: str,
    array: np.ndarray,
    *,
    expected_dtype: str | None = None,
    borrowable: bool = True,
    public: bool = True,
) -> _PendingBuffer:
    array = np.asarray(array)
    if array.dtype.hasobject:
        raise NumSimExecutionError(f"NumSim buffer {name!r} cannot use object dtype")
    dtype, itemsize = _dtype_name_and_size(name, array, expected_dtype)
    shape = tuple(int(value) for value in array.shape)
    packed_float4 = dtype == _PACKED_FLOAT4_DTYPE
    if packed_float4:
        shape = (array.size * 2,)
        byte_strides = ()
    else:
        byte_strides = tuple(int(value) for value in array.strides)

    owner, root_array = _array_owner(array)
    host_low, host_high = _byte_bounds(
        _array_pointer(array),
        tuple(int(value) for value in array.shape),
        tuple(int(value) for value in array.strides),
        int(array.dtype.itemsize),
    )
    origin = _array_pointer(array)
    root_origin = _array_pointer(root_array)
    root_low, root_high = _byte_bounds(
        root_origin,
        tuple(int(value) for value in root_array.shape),
        tuple(int(value) for value in root_array.strides),
        int(root_array.dtype.itemsize),
    )
    low, high = (
        (origin, origin + array.nbytes)
        if packed_float4
        else _byte_bounds(origin, shape, byte_strides, itemsize)
    )
    if low < root_low or high > root_high:
        raise NumSimExecutionError(
            f"NumSim buffer {name!r} byte range [{low}, {high}) escapes its "
            f"NumPy backing range [{root_low}, {root_high})"
        )
    backing_low, backing_high = root_low, root_high

    return _PendingBuffer(
        name=name,
        array=array,
        dtype=dtype,
        itemsize=itemsize,
        shape=shape,
        byte_strides=byte_strides,
        origin=origin,
        low=low,
        high=high,
        owner=owner,
        root_array=root_array,
        backing_low=backing_low,
        backing_high=backing_high,
        host_low=host_low,
        host_high=host_high,
        borrowable=borrowable,
        public=public,
    )


def _ranges_overlap(left: _PendingBuffer, right: _PendingBuffer) -> bool:
    return left.backing_low < right.backing_high and right.backing_low < left.backing_high


def _group_pending_buffers(pending: list[_PendingBuffer]) -> list[list[_PendingBuffer]]:
    """Build physical allocation components from NumPy host connectivity."""

    parent = list(range(len(pending)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for left_index, left in enumerate(pending):
        for right_index in range(left_index + 1, len(pending)):
            right = pending[right_index]
            if left.owner is right.owner or _ranges_overlap(left, right):
                union(left_index, right_index)

    grouped: dict[int, list[_PendingBuffer]] = {}
    for index, item in enumerate(pending):
        grouped.setdefault(find(index), []).append(item)
    return list(grouped.values())


def _immutable_bytes_owner(group_items: list[_PendingBuffer], low: int, high: int) -> bytes | None:
    """Return an exact immutable allocation owner that needs no host snapshot."""

    owner = group_items[0].owner
    if not isinstance(owner, bytes) or len(owner) != high - low:
        return None
    if any(
        item.owner is not owner
        or item.array.flags.writeable
        or item.backing_low != low
        or item.backing_high != high
        for item in group_items
    ):
        return None
    return owner


def _borrowed_host_owners(
    group_items: list[_PendingBuffer], low: int, high: int
) -> tuple[Any, ...]:
    """Retain owners whose NumPy ranges cover one borrowed physical allocation."""

    covering = sorted(
        [
            (max(low, item.backing_low), min(high, item.backing_high), item)
            for item in group_items
            if item.borrowable and item.backing_low < high and low < item.backing_high
        ],
        key=lambda entry: (entry[0], entry[1]),
    )
    cursor = low
    retained: list[Any] = []
    seen: set[int] = set()
    for start, end, item in covering:
        if start > cursor:
            break
        if end <= cursor:
            continue
        cursor = end
        for owner in (item.owner, item.root_array):
            if id(owner) not in seen:
                retained.append(owner)
                seen.add(id(owner))
        if cursor >= high:
            return tuple(retained)
    return ()


def _normalize_scalar_value(raw: Any, dtype: str) -> int | float | bool:
    if isinstance(raw, np.generic):
        raw = raw.item()
    if dtype == "bool":
        if not isinstance(raw, bool):
            raise NumSimExecutionError(
                f"NumSim scalar dtype 'bool' requires a bool value, got {type(raw).__name__}"
            )
        return raw
    if dtype in _INTEGER_SCALAR_DTYPES:
        if isinstance(raw, bool) or not isinstance(raw, int):
            raise NumSimExecutionError(
                f"NumSim scalar dtype {dtype!r} requires an integer value, got {type(raw).__name__}"
            )
        bits, signed = _INTEGER_SCALAR_DTYPES[dtype]
        minimum = -(1 << (bits - 1)) if signed else 0
        maximum = (1 << (bits - 1)) - 1 if signed else (1 << bits) - 1
        if raw < minimum or raw > maximum:
            raise NumSimExecutionError(
                f"NumSim scalar value {raw} is outside {dtype} range [{minimum}, {maximum}]"
            )
        return int(raw)
    if dtype in _FLOAT_SCALAR_DTYPES:
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise NumSimExecutionError(
                f"NumSim scalar dtype {dtype!r} requires a numeric value, got {type(raw).__name__}"
            )
        result = float(raw)
        if math.isfinite(result):
            if dtype == "bfloat16":
                maximum = float(np.finfo(np.float32).max)
            else:
                maximum = float(np.finfo(np.dtype(dtype)).max)
            if abs(result) > maximum:
                raise NumSimExecutionError(
                    f"NumSim scalar value {result} is outside finite {dtype} range"
                )
        return _normalize_float_scalar(result, dtype)
    raise NumSimExecutionError(f"unsupported NumSim scalar dtype {dtype!r}")


def _infer_scalar(value: Any, *, expected_dtype: str | None = None) -> PreparedScalar:
    raw = value.item() if isinstance(value, np.generic) else value
    dtype = None
    if not isinstance(raw, (bool, int, float)):
        raise NumSimExecutionError(f"unsupported NumSim scalar value {type(raw).__name__}")
    if dtype is None:
        if isinstance(value, np.generic):
            dtype = str(value.dtype)
        elif expected_dtype is not None:
            dtype = expected_dtype
        elif isinstance(raw, bool):
            dtype = "bool"
        elif isinstance(raw, int):
            dtype = "int64"
        else:
            dtype = "float64"
    dtype = str(dtype)
    return PreparedScalar(_normalize_scalar_value(raw, dtype), dtype)


@dataclass(frozen=True)
class _DecodedTensorMap:
    byte_offset: int
    address: int
    required_byte_len: int
    global_shape: tuple[int, ...]
    global_strides: tuple[int, ...]
    box_shape: tuple[int, ...]
    element_strides: tuple[int, ...]
    dtype: str
    fp4_shared_layout: str | None
    swizzle: str | None
    inactive_swizzle_atomicity: int
    fill_mode: str | None
    interleave_bytes: int | None
    im2col: Im2col | None

    @property
    def physical_global_shape(self) -> tuple[int, ...]:
        """Interleaved dimension zero counts slices, not scalar elements."""
        if self.interleave_bytes is None:
            return self.global_shape
        element_bits = _tensor_map_element_bits(self.dtype)
        elements = self.interleave_bytes * 8 // element_bits
        return (self.global_shape[0] * elements, *self.global_shape[1:])


def _decode_tensor_maps(array: np.ndarray) -> tuple[_DecodedTensorMap, ...]:
    if not array.flags.c_contiguous:
        return ()
    flat = array.view(np.uint8).reshape(-1)
    result: list[_DecodedTensorMap] = []
    for offset in range(0, flat.size - _TENSOR_MAP_DESCRIPTOR_BYTES + 1, 128):
        image = flat[offset : offset + _TENSOR_MAP_DESCRIPTOR_BYTES]
        tag = int(image[63])
        if tag & ~0x58 not in _TENSOR_MAP_FORMAT_TAGS.values():
            continue
        payload_bytes = _tensor_map_payload_bytes(tag)
        if np.any(image[payload_bytes:]):
            continue
        flags = int(image[60])
        if flags & 0x80 != _TENSOR_MAP_FLAG_MAGIC or flags & (1 << 5) == 0:
            continue
        if int.from_bytes(image[8:16].tobytes(), "little") != 0:
            continue
        rank = int(image[59]) & 0b111
        dtype_code = int(image[59]) >> 3
        if rank == 0 or rank > 5 or dtype_code >= len(_TENSOR_MAP_DTYPES):
            continue
        global_shape: list[int] = []
        for axis in range(rank):
            start = 16 + axis * 4
            encoded = int.from_bytes(image[start : start + 4].tobytes(), "little")
            global_shape.append(2**32 if encoded == 0 else encoded)
        if any(
            int.from_bytes(image[16 + axis * 4 : 20 + axis * 4].tobytes(), "little") != 1
            for axis in range(rank, 5)
        ):
            continue
        global_strides: list[int] = []
        for pair in range(2):
            start = 36 + pair * 9
            packed = int.from_bytes(image[start : start + 9].tobytes(), "little")
            global_strides.extend(((packed & ((1 << 36) - 1)) << 4, (packed >> 36) << 4))
        if any(global_strides[rank - 1 :]):
            continue
        dtype = _TENSOR_MAP_DTYPES[dtype_code]
        element_bits = _tensor_map_element_bits(dtype)
        interleave_bytes = next(
            width for width, base_tag in _TENSOR_MAP_FORMAT_TAGS.items() if base_tag == tag & ~0x58
        )
        if interleave_bytes is not None and rank < 3:
            continue
        transfer_bits = interleave_bytes * 8 if interleave_bytes is not None else element_bits
        required_byte_len = (global_shape[0] * transfer_bits + 7) // 8
        for stride, dimension in zip(global_strides[: rank - 1], global_shape[1:], strict=True):
            required_byte_len += (dimension - 1) * stride
        address = int.from_bytes(image[0:8].tobytes(), "little")
        if address == 0:
            continue
        fp4_shared_layout = {0: None, 1: "align8_packed", 2: "align16_padded"}.get(flags & 0b11)
        if fp4_shared_layout is None and flags & 0b11:
            continue
        if (dtype == _PACKED_FLOAT4_DTYPE) != (fp4_shared_layout is not None):
            continue
        swizzle_code = ((flags >> 2) & 0b11) | ((flags >> 4) & 4)
        atomicity = ((tag >> 3) & 1) | ((tag >> 5) & 2)
        swizzle = None
        if swizzle_code:
            try:
                swizzle = next(
                    name for name, code in _TENSOR_MAP_SWIZZLES.items()
                    if code == (swizzle_code, atomicity)
                )
            except StopIteration:
                continue
        box_shape = tuple(int(image[54 + axis]) + 1 for axis in range(rank))
        im2col = None
        if tag & 0x10:
            if rank < 3 or np.any(image[56:59]) or np.any(image[77:80]) or int(image[76]) & ~7:
                continue
            box_shape = (box_shape[0], box_shape[1] + ((int(image[76]) & 3) << 8))
            wide = bool(int(image[76]) & 4)
            spatial_rank = 1 if wide else rank - 2
            corners = struct.unpack_from("<3h3h", image, 64)
            im2col = Im2col(corners[:spatial_rank], corners[3 : 3 + spatial_rank], wide)
        elif np.any(image[54 + rank : 59]):
            continue
        packed_element_strides = int.from_bytes(image[61:63].tobytes(), "little")
        if packed_element_strides >> (rank * 3):
            continue
        element_strides = tuple(
            ((packed_element_strides >> (axis * 3)) & 0b111) + 1 for axis in range(rank)
        )
        result.append(
            _DecodedTensorMap(
                byte_offset=offset,
                address=address,
                required_byte_len=required_byte_len,
                global_shape=tuple(global_shape),
                global_strides=tuple(global_strides[: rank - 1]),
                box_shape=box_shape,
                element_strides=element_strides,
                dtype=dtype,
                fp4_shared_layout=fp4_shared_layout,
                swizzle=swizzle,
                inactive_swizzle_atomicity=atomicity if not swizzle_code else 0,
                fill_mode="nan" if flags & (1 << 4) else None,
                interleave_bytes=interleave_bytes,
                im2col=im2col,
            )
        )
    return tuple(result)


def _tensor_map_base_array(descriptor: np.ndarray) -> np.ndarray:
    decoded = _decode_tensor_maps(np.asarray(descriptor))
    if np.asarray(descriptor).shape != (128,) or len(decoded) != 1:
        raise NumSimExecutionError("expected one uint8[128] TensorMap descriptor")
    tensor_map = decoded[0]
    raw = np.ctypeslib.as_array(
        (ctypes.c_uint8 * tensor_map.required_byte_len).from_address(tensor_map.address)
    )
    return _tensor_map_array_from_buffer(
        raw,
        data_offset=0,
        global_shape=tensor_map.physical_global_shape,
        global_strides=tensor_map.global_strides,
        dtype=tensor_map.dtype,
    )


def _tensor_map_owner_array(
    descriptor_array: np.ndarray, descriptor: _DecodedTensorMap
) -> np.ndarray | None:
    """Recover an optional owner without deriving TensorMap semantics from it."""

    base = getattr(descriptor_array, "_tensor_map_base", None)
    if not isinstance(base, np.ndarray):
        return None
    _owner, root_array = _array_owner(base)
    root_low, root_high = _byte_bounds(
        _array_pointer(root_array),
        tuple(int(value) for value in root_array.shape),
        tuple(int(value) for value in root_array.strides),
        int(root_array.dtype.itemsize),
    )
    if not (
        root_low <= descriptor.address
        and descriptor.address + descriptor.required_byte_len <= root_high
    ):
        return None
    return root_array


def _tensor_map_physical_dtype(dtype: str) -> np.dtype[Any]:
    return np.dtype(
        {
            _PACKED_FLOAT4_DTYPE: "uint8",
            "uint6": "uint8",
            "float8_e4m3fn": "uint8",
            "float8_e8m0fnu": "uint8",
            "bfloat16": "uint16",
            "tf32": "float32",
            "float32_ftz": "float32",
            "tf32_ftz": "float32",
            "uint32x2": "uint64",
        }.get(dtype, dtype)
    )


def _tensor_map_array_from_buffer(
    buffer: Any,
    *,
    data_offset: int,
    global_shape: tuple[int, ...],
    global_strides: tuple[int, ...],
    dtype: str,
) -> np.ndarray:
    """Build the physical NumPy view described by TensorMap bytes."""

    physical_dtype = _tensor_map_physical_dtype(dtype)
    bits = _tensor_map_element_bits(dtype)
    dimension_zero = (global_shape[0] * bits + 7) // 8 if bits < 8 else global_shape[0]
    coordinate_shape = (dimension_zero, *global_shape[1:])
    coordinate_strides = (int(physical_dtype.itemsize), *global_strides)
    return np.ndarray(
        shape=tuple(reversed(coordinate_shape)),
        dtype=physical_dtype,
        buffer=buffer,
        offset=data_offset,
        strides=tuple(reversed(coordinate_strides)),
    )


def prepare_bindings(
    inputs: dict[str, Any],
    *,
    expected_scalar_dtypes: dict[str, str] | None = None,
    expected_buffer_dtypes: dict[str, str] | None = None,
    expected_tensor_map_names: set[str] | frozenset[str] | None = None,
) -> PreparedBindings:
    """Normalize ndarray/scalar inputs and discover TensorMaps from descriptor bytes."""

    expected_scalar_dtypes = expected_scalar_dtypes or {}
    expected_buffer_dtypes = expected_buffer_dtypes or {}
    expected_tensor_map_names = frozenset(expected_tensor_map_names or ())
    pending: list[_PendingBuffer] = []
    scalars: dict[str, PreparedScalar] = {}
    decoded_by_name: dict[str, tuple[_DecodedTensorMap, ...]] = {}
    for name in sorted(inputs):
        value = inputs[name]
        if isinstance(value, np.ndarray):
            pending.append(
                _normalize_buffer(
                    name,
                    value,
                    expected_dtype=expected_buffer_dtypes.get(name),
                )
            )
            decoded_by_name[name] = _decode_tensor_maps(value)
            continue
        if name in expected_buffer_dtypes or name in expected_tensor_map_names:
            raise NumSimExecutionError(
                f"NumSim buffer input {name!r} must be a NumPy array, got {type(value).__name__}"
            )
        scalars[name] = _infer_scalar(value, expected_dtype=expected_scalar_dtypes.get(name))

    for name in expected_tensor_map_names:
        descriptors = decoded_by_name.get(name, ())
        array = inputs.get(name)
        if (
            not isinstance(array, np.ndarray)
            or array.dtype != np.dtype(np.uint8)
            or array.shape != (128,)
        ):
            raise NumSimExecutionError(
                f"NumSim TensorMap input {name!r} must be a uint8[128] descriptor array"
            )
        if len(descriptors) != 1:
            raise NumSimExecutionError(
                f"NumSim TensorMap input {name!r} does not contain one valid descriptor"
            )

    hidden_index = 0
    for name, descriptors in decoded_by_name.items():
        descriptor_array = inputs[name]
        assert isinstance(descriptor_array, np.ndarray)
        for descriptor in descriptors:
            carrier = _tensor_map_owner_array(descriptor_array, descriptor)
            borrowable = carrier is not None
            if carrier is None:
                carrier = np.ctypeslib.as_array(
                    (ctypes.c_uint8 * descriptor.required_byte_len).from_address(descriptor.address)
                )
            pending.append(
                _normalize_buffer(
                    f"__numsim_tensor_map_base__:{hidden_index}",
                    carrier,
                    borrowable=borrowable,
                    public=False,
                )
            )
            hidden_index += 1

    allocations: list[PreparedAllocation] = []
    host_allocations: list[_HostAllocation] = []
    buffers: dict[str, PreparedBuffer] = {}
    originals: dict[str, _OriginalBuffer] = {}
    for group_items in _group_pending_buffers(pending):
        allocation_index = len(allocations)
        low = min(item.low for item in group_items)
        high = max(item.high for item in group_items)
        host_writeable = any(item.array.flags.writeable for item in group_items)
        immutable_owner = _immutable_bytes_owner(group_items, low, high)
        borrowed_owners = _borrowed_host_owners(group_items, low, high) if host_writeable else ()
        data: bytes | memoryview
        if immutable_owner is not None:
            data = immutable_owner
        elif borrowed_owners:
            data = _host_byte_view(low, high - low)
        else:
            data = _snapshot_host_bytes(low, high - low)
        allocations.append(
            PreparedAllocation(
                data=data,
                validity=b"",
                label=f"numpy-address:[{low:#x},{high:#x})",
                host_address=low,
            )
        )
        host_allocations.append(
            _HostAllocation(
                implicit=True,
                address=low,
                host_writeable=host_writeable,
                snapshot_is_immutable_owner=immutable_owner is not None,
                owners=borrowed_owners,
            )
        )
        for item in group_items:
            if not item.public:
                continue
            buffers[item.name] = PreparedBuffer(
                allocation=allocation_index,
                data_offset=item.origin - low,
                dtype=item.dtype,
                itemsize=item.itemsize,
                shape=item.shape,
                byte_strides=item.byte_strides,
            )
            originals[item.name] = _OriginalBuffer(
                array=item.array,
                pointer=_array_pointer(item.array),
                dtype=item.array.dtype.str,
                shape=tuple(int(value) for value in item.array.shape),
                byte_strides=tuple(int(value) for value in item.array.strides),
                host_writeable=bool(item.array.flags.writeable),
                owner=item.owner,
                root_array=item.root_array,
                root_pointer=_array_pointer(item.root_array),
                root_dtype=item.root_array.dtype.str,
                root_shape=tuple(int(value) for value in item.root_array.shape),
                root_byte_strides=tuple(int(value) for value in item.root_array.strides),
                root_host_writeable=bool(item.root_array.flags.writeable),
            )

    def allocation_for(address: int, byte_len: int) -> tuple[int, int]:
        for index, allocation in enumerate(allocations):
            start = allocation.host_address
            if start <= address and address + byte_len <= start + allocation.byte_len:
                return index, address - start
        raise NumSimExecutionError(
            f"NumSim bound address range [{address:#x}, {address + byte_len:#x}) "
            "is absent from the prepared allocations"
        )

    descriptor_allocations: set[int] = set()
    descriptor_storage_allocations: set[int] = set()
    tensor_map_outputs: dict[str, PreparedTensorMapOutput] = {}
    for name, descriptors in decoded_by_name.items():
        if descriptors:
            descriptor_allocation = buffers[name].allocation
            descriptor_allocations.add(descriptor_allocation)
            descriptor_storage_allocations.add(descriptor_allocation)
        for descriptor in descriptors:
            base_allocation, _base_offset = allocation_for(
                descriptor.address, descriptor.required_byte_len
            )
            descriptor_allocations.add(base_allocation)
        if name not in expected_tensor_map_names:
            continue
        descriptor = descriptors[0]
        allocation, data_offset = allocation_for(descriptor.address, descriptor.required_byte_len)
        tensor_map_outputs[name] = PreparedTensorMapOutput(
            allocation=allocation,
            data_offset=data_offset,
            global_shape=descriptor.physical_global_shape,
            global_strides=descriptor.global_strides,
            dtype=descriptor.dtype,
        )
        descriptor_allocations.add(allocation)

    return PreparedBindings(
        allocations=tuple(allocations),
        buffers=buffers,
        tensor_map_outputs=tensor_map_outputs,
        descriptor_allocations=frozenset(descriptor_allocations),
        descriptor_storage_allocations=frozenset(descriptor_storage_allocations),
        scalars=scalars,
        _originals=originals,
        _host_allocations=tuple(host_allocations),
    )
