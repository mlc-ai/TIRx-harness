"""Decode the table-driven ``tirx.ptx.*`` Call ABI into named operands.

The TVM PTX dialect serializes syntax metadata after the instruction operands:

``[present operands..., instruction predicate?] [modifier tokens...] [marker]``

Only this module knows that wire layout.  PTX handlers consume
:class:`DecodedPtxCall`; non-PTX calls continue through the existing raw-Call
path.
"""

from __future__ import annotations

import json
from collections import namedtuple
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from tvm.backend.cuda.ptx.table import (
    TABLE,
    InstructionEntry,
    lanes_of,
    mods,
    operand_dtypes,
    operand_layout,
    operand_type,
)

from ..errors import UnsupportedTIRxError
from . import native_frontend


class PtxCallDecodeError(UnsupportedTIRxError):
    """Raised when a ``tirx.ptx.*`` Call does not match the target table ABI."""


class _SunkPtxLane:
    __slots__ = ()

    def __repr__(self) -> str:
        return "PTX_SINK"

    def __reduce__(self):
        return (_restore_ptx_sink, ())


def _restore_ptx_sink() -> Any:
    return PTX_SINK


PTX_SINK = _SunkPtxLane()


class _FrozenMapping(Mapping[str, Any]):
    """Small immutable mapping that survives pickling."""

    __slots__ = ("_items", "_values")

    def __init__(self, items: Any = ()) -> None:
        self._items = tuple(items.items() if isinstance(items, Mapping) else items)
        self._values = dict(self._items)
        if len(self._items) != len(self._values):
            raise ValueError("frozen PTX mappings require unique keys")

    def __getitem__(self, key: str) -> Any:
        return self._values[key]

    def __iter__(self):
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)

    def __reduce__(self):
        return (type(self), (self._items,))


def _build_schema_index() -> Mapping[str, InstructionEntry]:
    by_op_name: dict[str, InstructionEntry] = {}
    for table_name, entry in TABLE.items():
        if table_name != entry.name:
            raise RuntimeError(
                f"target TVM PTX table key {table_name!r} does not match entry {entry.name!r}"
            )
        if not entry.op_name.startswith("tirx.ptx."):
            raise RuntimeError(f"target TVM PTX entry has non-PTX op name {entry.op_name!r}")
        previous = by_op_name.get(entry.op_name)
        if previous is not None:
            raise RuntimeError(
                f"target TVM PTX table maps {entry.op_name!r} to both "
                f"{previous.name!r} and {entry.name!r}"
            )
        by_op_name[entry.op_name] = entry
    return MappingProxyType(by_op_name)


PTX_SCHEMA_BY_OP_NAME = _build_schema_index()


@dataclass(frozen=True)
class DecodedPtxCall:
    """One structurally validated PTX instruction call.

    ``operands`` contains every operand grouped by table slot name. A dynamic
    register group is one tuple, a discarded lane is represented by
    :data:`PTX_SINK`, and a table-owned literal is a one-element string tuple.
    The payload is self-contained; target-table objects never cross the decode
    boundary.
    """

    op_name: str
    operands: Mapping[str, tuple[Any, ...]]
    modifiers: Mapping[str, str]
    predicate: Any | None
    span: Any
    result_type: Any
    preserve_dst: bool = False

    def operand(self, name: str) -> tuple[Any, ...]:
        """Return one named operand group."""

        try:
            return self.operands[name]
        except KeyError as err:
            raise KeyError(f"{self.op_name} has no operand named {name!r}") from err

    def scalar_operand(self, name: str) -> Any:
        """Return a named one-lane operand, rejecting vector use by mistake."""

        values = self.operand(name)
        if len(values) != 1:
            raise ValueError(
                f"{self.op_name} operand {name!r} has {len(values)} lanes, expected one"
            )
        return values[0]

    def modifier(self, name: str) -> str:
        """Return one modifier token, with ``""`` denoting an omitted slot."""

        try:
            return self.modifiers[name]
        except KeyError as err:
            raise KeyError(f"{self.op_name} has no modifier slot named {name!r}") from err


def _op_name(call: Any) -> str:
    op = getattr(call, "op", None)
    return str(getattr(op, "name", op))


def _raise(call: Any, op_name: str, message: str) -> None:
    span = getattr(call, "span", None)
    location = f" at {span}" if span is not None else ""
    raise PtxCallDecodeError(
        f"{op_name}{location}: {message}",
        unsupported=(op_name,),
        source_span=span,
    )


def _string_imm(call: Any, op_name: str, value: Any, token: str | None, *, field: str) -> str:
    if token is None:
        _raise(call, op_name, f"{field} must be a StringImm, got {type(value).__name__}")
    return str(token)


def _parse_marker(
    call: Any, op_name: str, marker: str
) -> tuple[bool, bool, frozenset[int], frozenset[int]]:
    flags = marker.split(",") if marker else []
    if any(not flag for flag in flags):
        _raise(call, op_name, f"marker {marker!r} contains an empty flag")
    if len(flags) != len(set(flags)):
        _raise(call, op_name, f"marker {marker!r} contains duplicate flags")

    predicated = False
    preserve_dst = False
    pred_registers: set[int] = set()
    sinks: set[int] = set()
    for flag in flags:
        if flag == "pred":
            predicated = True
            continue
        if flag == "keep":
            preserve_dst = True
            continue
        if len(flag) < 2 or flag[0] not in {"p", "s"} or not flag[1:].isdigit():
            _raise(call, op_name, f"marker contains unknown flag {flag!r}")
        position = int(flag[1:])
        (pred_registers if flag[0] == "p" else sinks).add(position)
    if preserve_dst and not predicated:
        _raise(call, op_name, "destination-preservation marker requires an instruction predicate")
    return predicated, preserve_dst, frozenset(pred_registers), frozenset(sinks)


def decode_ptx_call(call: Any) -> DecodedPtxCall:
    """Decode one target-TVM table-driven PTX call.

    The function intentionally rejects every non-PTX operation.  Syntax facts
    come from the target TVM table; support policy and runtime semantics remain
    owned by the registered TIRx Tools handler.
    """

    if type(call).__name__ != "Call":
        raise TypeError(f"decode_ptx_call expects a TIRx Call, got {type(call).__name__}")

    op_name = _op_name(call)
    if not op_name.startswith("tirx.ptx."):
        _raise(call, op_name, "PTX decoding only applies to tirx.ptx.* operations")
    entry = PTX_SCHEMA_BY_OP_NAME.get(op_name)
    if entry is None:
        _raise(call, op_name, "operation is not present in the target TVM PTX table")

    op_name, args, text_tokens, span, result_type = native_frontend.ptx_call_parts(
        call, len(entry.slots)
    )
    args = tuple(args)
    metadata_count = len(entry.slots) + 1
    if len(args) < metadata_count:
        _raise(
            call,
            op_name,
            f"expected {len(entry.slots)} modifier token(s) and one marker, got {len(args)} total argument(s)",
        )

    marker = _string_imm(call, op_name, args[-1], text_tokens[-1], field="trailing marker")
    token_nodes = args[-metadata_count:-1]
    tokens = tuple(
        _string_imm(call, op_name, node, token, field=f"modifier slot {slot.name!r}")
        for slot, node, token in zip(entry.slots, token_nodes, text_tokens[:-1])
    )
    modifier_map = mods(entry, tokens)
    for slot, token in zip(entry.slots, tokens):
        if not token and not slot.optional:
            _raise(call, op_name, f"required modifier slot {slot.name!r} is empty")
        if token and token not in slot.choices:
            _raise(
                call,
                op_name,
                f"modifier {slot.name!r} has token {token!r}, expected one of {slot.choices}",
            )
    if entry.check is not None:
        error = entry.check(modifier_map)
        if error:
            _raise(call, op_name, f"illegal modifier combination: {error}")

    layout = operand_layout(entry, modifier_map)
    logical_lane_count = sum(lanes for _, _, lanes in layout)
    predicated, preserve_dst, pred_registers, sinks = _parse_marker(call, op_name, marker)
    if preserve_dst and not entry.has_dst:
        _raise(call, op_name, "destination-preservation marker requires a destination")

    for label, positions in (("predicate-register", pred_registers), ("sink", sinks)):
        invalid = sorted(position for position in positions if position >= logical_lane_count)
        if invalid:
            _raise(
                call,
                op_name,
                f"{label} marker position(s) {invalid} outside operand lane range [0, {logical_lane_count})",
            )

    expected_pred_registers = frozenset(
        first
        for slot, first, lanes in layout
        if lanes
        and slot.kind == "reg"
        and slot.rw == "r"
        and operand_type(slot, modifier_map) == "pred"
    )
    if pred_registers != expected_pred_registers:
        _raise(
            call,
            op_name,
            f"predicate-register marker positions {sorted(pred_registers)} do not match schema "
            f"positions {sorted(expected_pred_registers)}",
        )
    for slot, first, lanes in layout:
        sunk_lanes = [lane for lane in range(lanes) if first + lane in sinks]
        if not sunk_lanes:
            continue
        sinkable = slot.sinkable(modifier_map) if callable(slot.sinkable) else slot.sinkable
        if not (sinkable and slot.kind == "reg"):
            _raise(call, op_name, f"operand {slot.name!r} is not sinkable for these modifiers")
        if len(sunk_lanes) == lanes:
            _raise(call, op_name, f"every lane of operand {slot.name!r} is sunk")

    serialized_operands = args[:-metadata_count]
    present_operand_count = logical_lane_count - len(sinks)
    expected_serialized_count = present_operand_count + int(predicated)
    if len(serialized_operands) != expected_serialized_count:
        _raise(
            call,
            op_name,
            f"schema requires {present_operand_count} present operand lane(s)"
            f"{' plus one instruction predicate' if predicated else ''}, got {len(serialized_operands)}",
        )

    predicate = serialized_operands[-1] if predicated else None
    present_values = iter(serialized_operands[:-1] if predicated else serialized_operands)
    operand_bindings: dict[str, tuple[Any, ...]] = {}
    layout_rows = iter(layout)
    for slot in entry.operands:
        if slot.name in operand_bindings:
            _raise(call, op_name, f"schema repeats operand name {slot.name!r}")
        if slot.kind == "imm" and slot.literal is not None:
            operand_bindings[slot.name] = (slot.literal,)
            continue
        try:
            layout_slot, first, lanes = next(layout_rows)
        except StopIteration:  # pragma: no cover - target-table invariant
            _raise(call, op_name, "internal decoder error: operand layout ended early")
        if layout_slot is not slot:  # pragma: no cover - target-table invariant
            _raise(call, op_name, "internal decoder error: operand layout order mismatch")
        values = tuple(
            PTX_SINK if first + lane in sinks else next(present_values) for lane in range(lanes)
        )
        operand_bindings[slot.name] = values
    try:
        next(layout_rows)
    except StopIteration:
        pass
    else:  # pragma: no cover - target-table invariant
        _raise(call, op_name, "internal decoder error: operand layout has extra rows")
    try:
        next(present_values)
    except StopIteration:
        pass
    else:  # pragma: no cover - guarded by the exact count check above
        _raise(call, op_name, "internal decoder error: unconsumed operand values")

    return DecodedPtxCall(
        op_name=op_name,
        operands=_FrozenMapping(operand_bindings),
        modifiers=_FrozenMapping(modifier_map),
        predicate=predicate,
        preserve_dst=preserve_dst,
        span=span,
        result_type=result_type,
    )


def _operand_payload(
    call: Any, decoded: DecodedPtxCall, call_payload: type, operand_payload: type
) -> dict[str, Any]:
    entry = PTX_SCHEMA_BY_OP_NAME[decoded.op_name]
    modifiers = dict(decoded.modifiers)
    arguments = tuple(call.args)

    def position(value: Any) -> int:
        for index, argument in enumerate(arguments):
            if argument.same_as(value):
                return index
        raise RuntimeError(f"{decoded.op_name} decoded operand is not a call argument")

    operands = []
    for slot in entry.operands:
        register_type = ""
        dtypes = []
        if slot.kind == "reg":
            try:
                register_type = operand_type(slot, modifiers)
                dtypes = list(operand_dtypes(slot, modifiers))
            except KeyError:
                # The target table leaves this slot's register contract untyped.
                pass
        values = []
        if not (slot.kind == "imm" and slot.literal is not None):
            values = [
                -1 if value is PTX_SINK else position(value) for value in decoded.operand(slot.name)
            ]
        operands.append(
            operand_payload(
                name=slot.name,
                kind=slot.kind,
                rw=slot.rw,
                lanes=lanes_of(slot, modifiers),
                allow_imm_offset=bool(slot.allow_imm_offset),
                literal=None if slot.literal is None else str(slot.literal),
                operand_type=register_type,
                dtypes=dtypes,
                values=values,
            )._asdict()
        )
    return call_payload(
        op_name=decoded.op_name,
        modifiers=[[name, token] for name, token in decoded.modifiers.items()],
        predicate=None if decoded.predicate is None else position(decoded.predicate),
        preserve_dst=decoded.preserve_dst,
        result_type=str(decoded.result_type),
        operands=operands,
    )._asdict()


NATIVE_PTX_DECODER = "numsim.frontend.decode_ptx_call"


def register_native_ptx_decoder(payload_schema: Any) -> None:
    """Register the target-table decoder using the native payload field definitions."""

    import tvm_ffi

    call_fields = tuple(map(str, payload_schema["call"]))
    call_payload = namedtuple("PtxCallPayload", call_fields, defaults=(None,) * len(call_fields))
    operand_payload = namedtuple("PtxOperandPayload", map(str, payload_schema["operand"]))

    def decode_payload(node: Any) -> str:
        try:
            decoded = decode_ptx_call(node)
        except PtxCallDecodeError as error:
            payload = call_payload(op_name=_op_name(node), error=str(error))._asdict()
        else:
            payload = _operand_payload(node, decoded, call_payload, operand_payload)
        return json.dumps(payload, separators=(",", ":"))

    tvm_ffi.register_global_func(NATIVE_PTX_DECODER, decode_payload, override=True)

    def build_call(spelling: str, operands: Any) -> Any:
        from tvm.backend.cuda.ptx import PTXNamespace

        try:
            call = PTXNamespace()[str(spelling)](*operands)
        except (KeyError, TypeError, ValueError) as error:
            return tvm_ffi.convert((None, str(error)))
        return tvm_ffi.convert((call, ""))

    tvm_ffi.register_global_func("numsim.frontend.build_ptx_call", build_call, override=True)


__all__ = [
    "DecodedPtxCall",
    "PTX_SCHEMA_BY_OP_NAME",
    "PTX_SINK",
    "PtxCallDecodeError",
    "decode_ptx_call",
    "register_native_ptx_decoder",
]
