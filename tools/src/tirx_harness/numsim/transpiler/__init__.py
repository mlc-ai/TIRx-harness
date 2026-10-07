"""TIRx-to-Rust transpiler implementation."""

from .frontend import ModuleSpec, PrimFuncSpec, analyze

__all__ = ["ModuleSpec", "PrimFuncSpec", "analyze"]
