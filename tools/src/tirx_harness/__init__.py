"""Installable tooling for TIRx kernel development."""

from __future__ import annotations

from importlib.metadata import version
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tvm.tir import PrimFunc

__version__ = version("tirx-harness")

__all__ = ["racecheck", "synccheck"]


def synccheck(kernel: PrimFunc, inputs: dict | list[dict] | None = None):
    """Run native synchronization analysis for one concrete TIRx invocation.

    ``inputs`` may be a list of per-rank dicts for a multi-GPU launch; bind
    multicast addresses with ``tirx_harness.numsim.MulticastWindow``.
    """
    from .numsim.checkers import synccheck as _synccheck

    return _synccheck(kernel, inputs)


def racecheck(kernel: PrimFunc, inputs: dict | list[dict] | None = None):
    """Run native data-race analysis for one concrete TIRx invocation.

    ``inputs`` may be a list of per-rank dicts for a multi-GPU launch; bind
    multicast addresses with ``tirx_harness.numsim.MulticastWindow``.
    """
    from .numsim.checkers import racecheck as _racecheck

    return _racecheck(kernel, inputs)
