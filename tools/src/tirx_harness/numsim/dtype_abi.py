"""Storage-only scalar and packed-vector ABI classification for NumSim.

The scalar table records physical bit widths only. Membership does not imply
that NumSim has a numerical codec or that any particular instruction accepts
the dtype. Vector entries likewise describe byte transport, not arithmetic.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

# The same registry is embedded in the Rust frontend at build time. Storage
# classification deliberately excludes sub-byte types and ignores capabilities.
_REGISTRY = json.loads(Path(__file__).with_name("dtype_registry.json").read_text())
SCALAR_DTYPE_BITS = {
    name: entry["bits"] for name, entry in _REGISTRY["types"].items() if entry["bits"] >= 8
}

_VECTOR_DTYPE = re.compile(r"^(?P<element>.+)x(?P<lanes>[1-9][0-9]*)$")
_PACKED_VECTOR_WIDTHS = frozenset({16, 32, 64, 128})


@dataclass(frozen=True)
class VectorDTypeABI:
    dtype: str
    element_dtype: str
    lanes: int
    element_bits: int
    total_bits: int

    @property
    def itemsize(self) -> int:
        return self.total_bits // 8


def vector_dtype_abi(dtype: str) -> VectorDTypeABI | None:
    """Return the fixed-width storage ABI for one ordinary vector dtype."""

    match = _VECTOR_DTYPE.fullmatch(dtype)
    if match is None:
        return None
    element_dtype = match.group("element")
    if element_dtype == "bool":
        return None
    element_bits = SCALAR_DTYPE_BITS.get(element_dtype)
    if element_bits is None:
        return None
    lanes = int(match.group("lanes"))
    if lanes <= 1:
        return None
    total_bits = element_bits * lanes
    if total_bits not in _PACKED_VECTOR_WIDTHS:
        return None
    return VectorDTypeABI(dtype, element_dtype, lanes, element_bits, total_bits)


def vector_dtype_abis() -> tuple[VectorDTypeABI, ...]:
    """Every vector dtype ``vector_dtype_abi`` accepts, in dtype order."""

    candidates = (
        f"{element_dtype}x{lanes}"
        for element_dtype, element_bits in SCALAR_DTYPE_BITS.items()
        for lanes in range(2, max(_PACKED_VECTOR_WIDTHS) // element_bits + 1)
    )
    return tuple(sorted(filter(None, map(vector_dtype_abi, candidates)), key=lambda abi: abi.dtype))


def dtype_itemsize(dtype: str) -> int | None:
    """Return the ordinary byte-addressed itemsize, if NumSim can represent it."""

    scalar_bits = SCALAR_DTYPE_BITS.get(dtype)
    if scalar_bits is not None:
        return scalar_bits // 8
    vector = vector_dtype_abi(dtype)
    return None if vector is None else vector.itemsize


__all__ = [
    "SCALAR_DTYPE_BITS",
    "VectorDTypeABI",
    "dtype_itemsize",
    "vector_dtype_abi",
    "vector_dtype_abis",
]
