from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from tirx_harness.numsim.transpiler.cache import PYO3_VERSION, rust_tool
from tests.numsim.support.paths import ENGINE_ROOT

MODULE_NAME = "_numsim_numpy_backend_probe"


def _cargo_manifest() -> str:
    return f'''[package]
name = "numsim-numpy-backend-probe"
version = "0.0.0"
edition = "2024"
publish = false

[lib]
name = "{MODULE_NAME}"
crate-type = ["cdylib"]

[dependencies]
numsim-engine = {{ path = {json.dumps(str(ENGINE_ROOT))}, features = ["python"] }}
pyo3 = {{ version = "={PYO3_VERSION}", features = ["extension-module"] }}
'''


PROBE_SOURCE = r"""
use numsim_engine::artifact_support::{f32_to_bf16_bits, f32_to_fp16_bits};
use pyo3::prelude::*;
use pyo3::types::PyModule;

type ProbeResult = (Vec<u16>, Vec<u16>);

#[pyfunction]
fn exercise() -> PyResult<ProbeResult> {
    let values = [1.0, -2.0, 0.5];
    Ok((
        values.into_iter().map(f32_to_fp16_bits).collect(),
        values.into_iter().map(f32_to_bf16_bits).collect(),
    ))
}

#[pymodule]
fn _numsim_numpy_backend_probe(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(exercise, module)?)?;
    Ok(())
}
"""


def _build_probe(tmp_path: Path) -> Path:
    source_dir = tmp_path / "src"
    source_dir.mkdir()
    (tmp_path / "Cargo.toml").write_text(_cargo_manifest())
    (source_dir / "lib.rs").write_text(PROBE_SOURCE)

    env = os.environ.copy()
    env["PYO3_PYTHON"] = sys.executable
    env["PYO3_USE_ABI3_FORWARD_COMPATIBILITY"] = "1"
    completed = subprocess.run(
        [rust_tool("cargo"), "build", "--release", "--message-format=json"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        pytest.fail(f"NumPy backend probe failed to build:\n{completed.stdout}{completed.stderr}")

    libraries: list[Path] = []
    for line in completed.stdout.splitlines():
        try:
            message = json.loads(line)
        except json.JSONDecodeError:
            continue
        target = message.get("target", {})
        if message.get("reason") != "compiler-artifact" or target.get("name") != MODULE_NAME:
            continue
        libraries.extend(
            Path(filename) for filename in message.get("filenames", []) if filename.endswith(".so")
        )
    assert len(libraries) == 1, libraries
    return libraries[0]


def test_numeric_facade_loads_from_a_pyo3_extension(tmp_path):
    library = _build_probe(tmp_path)
    spec = importlib.util.spec_from_file_location(MODULE_NAME, library)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    fp16, bf16 = module.exercise()
    assert fp16 == [0x3C00, 0xC000, 0x3800]
    assert bf16 == [0x3F80, 0xC000, 0x3F00]
