from pathlib import Path

from tirx_harness.numsim import abi
from tirx_harness.numsim.abi import NUMSIM_ABI_VERSION, abi_metadata
from tirx_harness.numsim.transpiler.frontend import analyze
from tirx_harness.numsim.transpiler.artifact_template import emit_rust_module

from tests.numsim.support.kernels import lane_add


def test_numsim_uses_one_artifact_abi_version():
    assert abi_metadata() == {"numsim_abi_version": NUMSIM_ABI_VERSION}
    engine = Path(abi.__file__).parent / "engine-rs" / "src" / "lib.rs"
    assert f"const NUMSIM_ABI_VERSION: u32 = {NUMSIM_ABI_VERSION};" in engine.read_text()


def test_module_manifest_has_no_parallel_semantic_or_boundary_versions():
    spec = analyze(lane_add)
    manifest = spec.to_manifest()

    assert "semantic_contract_ids" not in manifest
    assert "semantic_contract_digest" not in manifest
    source = emit_rust_module(spec, lane_add)
    assert f"const NUMSIM_ABI_VERSION: u32 = {NUMSIM_ABI_VERSION};" in source
    for stale in (
        "SEMANTIC_MODEL_EPOCH",
        "CODEGEN_ABI_EPOCH",
        "ENGINE_FACADE_MAJOR",
        "ARTIFACT_HOST_ABI_VERSION",
        "BINDING_ABI_VERSION",
    ):
        assert stale not in source
