"""Canonical host binding names for single- and multi-kernel artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

BindingKind = Literal["buffer", "pointer", "scalar", "tensor_map"]


class HostAbiError(ValueError):
    """The analyzed module cannot be represented by the public host ABI."""


@dataclass(frozen=True)
class HostBindingSlot:
    kernel_index: int
    kind: BindingKind
    local_name: str
    canonical_name: str
    local_aliases: tuple[str, ...]
    dtype: str | None = None

    @property
    def output_capable(self) -> bool:
        return self.kind != "scalar"


@dataclass(frozen=True)
class ImplicitTensorMapSlot:
    """One TensorMap binding derived from an ordinary host buffer binding."""

    canonical_name: str
    base_canonical_name: str


class HostAbiContract:
    """Exact binding slots and deterministic aliases for one ModuleSpec.

    ``host_abi.rs`` names every slot and rejects a module the public ABI cannot
    spell; this is the Python view of those facts.
    """

    def __init__(self, facts: dict[str, Any]) -> None:
        self.kernel_count = facts["kernel_count"]
        self.slots = tuple(
            HostBindingSlot(
                kernel_index=slot["kernel_index"],
                kind=slot["kind"],
                local_name=slot["local_name"],
                canonical_name=slot["canonical_name"],
                local_aliases=tuple(slot["local_aliases"]),
                dtype=slot["dtype"],
            )
            for slot in facts["slots"]
        )
        self.implicit_tensor_maps = tuple(
            ImplicitTensorMapSlot(
                canonical_name=item["canonical_name"],
                base_canonical_name=item["base_canonical_name"],
            )
            for item in facts["implicit_tensor_maps"]
        )
        self.implicit_tensor_map_names = frozenset(
            item.canonical_name for item in self.implicit_tensor_maps
        )
        self._by_canonical = {slot.canonical_name: slot for slot in self.slots}
        self._alias_targets = {
            alias: tuple(targets) for alias, targets in facts["alias_targets"].items()
        }

    @property
    def unique_aliases(self) -> dict[str, str]:
        return {
            alias: targets[0] for alias, targets in self._alias_targets.items() if len(targets) == 1
        }

    @property
    def ambiguous_aliases(self) -> dict[str, tuple[str, ...]]:
        return {
            alias: targets for alias, targets in self._alias_targets.items() if len(targets) > 1
        }

    @property
    def known_aliases(self) -> tuple[str, ...]:
        return tuple(sorted(self._alias_targets))

    @property
    def scalar_dtypes(self) -> dict[str, str]:
        return {
            slot.canonical_name: slot.dtype
            for slot in self.slots
            if slot.kind == "scalar" and slot.dtype is not None
        }

    @property
    def buffer_dtypes(self) -> dict[str, str]:
        return {
            slot.canonical_name: slot.dtype
            for slot in self.slots
            if slot.kind in {"buffer", "pointer"} and slot.dtype is not None
        }

    @property
    def tensor_map_names(self) -> frozenset[str]:
        return frozenset(
            slot.canonical_name
            for slot in self.slots
            if slot.kind == "tensor_map"
            and slot.canonical_name not in self.implicit_tensor_map_names
        )

    def bound_tensor_map_names(self, canonical_inputs: dict[str, Any]) -> frozenset[str]:
        """Return explicit parameters plus caller-provided implicit-map overrides."""

        return self.tensor_map_names | (self.implicit_tensor_map_names & canonical_inputs.keys())

    @property
    def output_binding_names(self) -> frozenset[str]:
        return frozenset(slot.canonical_name for slot in self.slots if slot.output_capable)

    def slot(self, canonical_name: str) -> HostBindingSlot:
        try:
            return self._by_canonical[canonical_name]
        except KeyError as error:
            raise HostAbiError(f"unknown canonical host binding {canonical_name!r}") from error


def build_host_abi(spec: Any) -> HostAbiContract:
    """The host ABI the native frontend derived for ``spec``."""

    facts = spec.host_abi
    if "error" in facts:
        raise HostAbiError(facts["error"])
    return HostAbiContract(facts)


__all__ = [
    "BindingKind",
    "HostAbiContract",
    "HostAbiError",
    "HostBindingSlot",
    "ImplicitTensorMapSlot",
    "build_host_abi",
]
