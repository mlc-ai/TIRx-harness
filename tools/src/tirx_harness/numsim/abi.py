"""Compatibility version shared by every NumSim artifact boundary."""

from __future__ import annotations

NUMSIM_ABI_VERSION = 40


def abi_metadata() -> dict[str, int]:
    """Return the compatibility metadata embedded in a native artifact."""

    return {"numsim_abi_version": NUMSIM_ABI_VERSION}


__all__ = ["NUMSIM_ABI_VERSION", "abi_metadata"]
