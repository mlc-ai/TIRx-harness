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
    "UnmodeledTIRxFormError",
    "UnsupportedTIRxError",
    "compare",
    "dump_rust",
    "dump_semantic_manifest",
    "run_case",
    "transpile",
]
