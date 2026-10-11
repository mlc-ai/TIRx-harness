"""NumSim: native-Rust numerical simulation for specialized TIRx kernels."""

from .api import (
    CompiledModule,
    CoverageBounds,
    Engine,
    NativeAnalysisResult,
    NumSimResult,
    ResourceLimits,
    compare,
    dump_rust,
    dump_semantic_manifest,
    run_case,
    transpile,
)
from .bindings import MulticastWindow, SymmetricBuffer, rank_binding_name
from .cases import (
    ComparisonRegion,
    ComparisonSpec,
    ExecutionAssumptions,
    NumSimCase,
    Im2col,
    TensorMap,
)
from .errors import (
    NumSimBuildError,
    NumSimError,
    NumSimExecutionError,
    UnmodeledTIRxFormError,
    UnsupportedTIRxError,
)

__all__ = [
    "ComparisonRegion",
    "ComparisonSpec",
    "CompiledModule",
    "CoverageBounds",
    "Engine",
    "ExecutionAssumptions",
    "NativeAnalysisResult",
    "NumSimBuildError",
    "NumSimCase",
    "NumSimError",
    "NumSimExecutionError",
    "NumSimResult",
    "ResourceLimits",
    "TensorMap",
    "Im2col",
    "MulticastWindow",
    "SymmetricBuffer",
    "UnmodeledTIRxFormError",
    "UnsupportedTIRxError",
    "compare",
    "dump_rust",
    "dump_semantic_manifest",
    "rank_binding_name",
    "run_case",
    "transpile",
]
