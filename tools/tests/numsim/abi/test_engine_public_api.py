from tirx_harness.numsim.transpiler.frontend import analyze
from tirx_harness.numsim.transpiler.artifact_template import emit_rust_module

from tests.numsim.support.kernels import lane_add


def test_generated_rust_uses_only_the_declared_engine_boundaries():
    source = emit_rust_module(analyze(lane_add), lane_add)
    imports = "\n".join(
        (
            "// numsim-engine-imports:begin",
            "use numsim_engine::artifact_support::*;",
            "use numsim_engine::abi::v2;",
            "// numsim-engine-imports:end",
        )
    )

    assert source.count(imports) == 1
    assert "numsim_engine::" not in source.replace(imports, "")
