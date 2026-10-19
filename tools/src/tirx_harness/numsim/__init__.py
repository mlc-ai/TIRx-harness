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
    Im2col,
    NumSimCase,
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
    "Im2col",
    "NativeAnalysisResult",
    "NumSimBuildError",
    "NumSimCase",
    "NumSimError",
    "NumSimExecutionError",
    "NumSimResult",
    "ResourceLimits",
    "TensorMap",
    "UnmodeledTIRxFormError",
    "UnsupportedTIRxError",
    "compare",
    "dump_rust",
    "dump_semantic_manifest",
    "run_case",
    "transpile",
]
