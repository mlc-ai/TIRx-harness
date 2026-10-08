"""Import canonical kernels directly from the installed ``tirx_kernels`` package."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from types import ModuleType
from typing import Any

import tirx_kernels
from tirx_kernels.registry import load_kernel


def config_params(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return kernel/corpus parameters without the human-readable case label."""
    return {name: value for name, value in config.items() if name != "label"}


def tirx_kernels_root() -> Path:
    """Return the canonical source directory owned by the imported package."""
    package_file = getattr(tirx_kernels, "__file__", None)
    if package_file is None:
        raise RuntimeError("tirx_kernels must be imported from a regular package")
    root = Path(package_file).resolve().parent
    if not root.is_dir():
        raise FileNotFoundError(f"tirx_kernels package root is not a directory: {root}")
    return root


def load_tirx_kernel(name: str) -> ModuleType:
    """Import one canonical kernel by its public registry identity."""
    if not isinstance(name, str) or not name.isidentifier():
        raise ValueError(f"invalid tirx_kernels registry name: {name!r}")
    return load_kernel(name, strict=True)


__all__ = ["config_params", "load_tirx_kernel", "tirx_kernels_root"]
