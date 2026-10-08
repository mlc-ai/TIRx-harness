from __future__ import annotations

from tirx_harness.numsim.abi import NUMSIM_ABI_VERSION
from tirx_harness.numsim.transpiler.support_matrix import (
    render_engine_support_matrix,
    write_engine_support_matrix,
)
from tests.numsim.support.paths import ENGINE_ROOT

_SUPPORT_MATRIX = ENGINE_ROOT / "SUPPORTED_OPS.md"
_ENGINE_LIB = ENGINE_ROOT / "src" / "lib.rs"


def test_engine_support_matrix_matches_declarative_registries():
    assert _SUPPORT_MATRIX.read_text() == render_engine_support_matrix()


def test_engine_support_matrix_writer_is_deterministic_and_then_a_noop(tmp_path):
    path = tmp_path / "SUPPORTED_OPS.md"

    assert write_engine_support_matrix(path)
    assert path.read_text(encoding="utf-8") == render_engine_support_matrix()
    assert not write_engine_support_matrix(path)


def test_python_and_rust_abi_versions_match():
    source = _ENGINE_LIB.read_text()
    assert f"pub(crate) const NUMSIM_ABI_VERSION: u32 = {NUMSIM_ABI_VERSION};" in source
