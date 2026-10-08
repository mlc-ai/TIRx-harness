from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tomllib
import zipfile
from pathlib import Path

import tvm
import tvm_ffi

from tests.numsim.support.paths import REPO_ROOT, TOOLS_ROOT


def test_wheel_contains_the_runtime_rust_engine(tmp_path):
    project_root = REPO_ROOT
    source_package = TOOLS_ROOT / "src" / "tirx_harness" / "numsim"
    tvm_python_root = Path(tvm.__file__).resolve().parent.parent
    tvm_ffi_root = Path(tvm_ffi.__file__).resolve().parent.parent
    env = os.environ.copy()
    env["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
    sdist_dir = tmp_path / "sdist"
    sdist = subprocess.run(
        [sys.executable, "setup.py", "sdist", "--dist-dir", str(sdist_dir)],
        cwd=project_root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert sdist.returncode == 0, sdist.stdout + sdist.stderr
    unpacked = tmp_path / "source"
    shutil.unpack_archive(str(next(sdist_dir.glob("*.tar.gz"))), unpacked)
    project = next(unpacked.iterdir())
    project_metadata = tomllib.loads((project / "pyproject.toml").read_text())["project"]
    assert project_metadata["name"] == "tirx-harness"
    assert not (project / "tools" / "setup.py").exists()

    # The archive must preserve the pinned upstream bindings byte for byte.
    original_bindings = project_root / "thirdparty" / "tvm-rust-ext" / "src"
    bundled_bindings = project / "tools" / "frontend-rs" / "thirdparty" / "tvm-rust-ext" / "src"
    for original in original_bindings.rglob("*.rs"):
        bundled = bundled_bindings / original.relative_to(original_bindings)
        assert bundled.read_bytes() == original.read_bytes(), original.name

    wheel_dir = tmp_path / "wheel"
    wheel_dir.mkdir()
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            ".",
            "--no-deps",
            "--no-build-isolation",
            "--wheel-dir",
            str(wheel_dir),
        ],
        cwd=project,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr

    wheels = list(wheel_dir.glob("*.whl"))
    assert len(wheels) == 1
    with zipfile.ZipFile(wheels[0]) as archive:
        packaged = set(archive.namelist())
        metadata_name = next(name for name in packaged if name.endswith(".dist-info/METADATA"))
        metadata = archive.read(metadata_name).decode()
        native_identity = archive.read("tirx_harness/numsim/_tvm_rust_ext.identity").decode().strip()
        assert native_identity
        assert "tirx_harness/numsim/dtype_registry.json" in packaged

    engine_root = source_package / "engine-rs"
    expected = {
        f"tirx_harness/numsim/engine-rs/{path.relative_to(engine_root).as_posix()}"
        for path in engine_root.rglob("*")
        if path.is_file()
        and "target" not in path.parts
        and "__pycache__" not in path.parts
        and path.suffix != ".pyc"
        and path.name != ".gitignore"
        and path.relative_to(engine_root).parts[0] != "tests"
    }
    missing = sorted(expected - packaged)
    assert not missing, f"wheel omitted NumSim engine files: {missing}"
    packaged_engine_tests = sorted(
        name for name in packaged if name.startswith("tirx_harness/numsim/engine-rs/tests/")
    )
    assert not packaged_engine_tests, (
        f"wheel must not include source-tree-only Rust conformance tests: {packaged_engine_tests}"
    )
    assert "Name: tirx-harness\n" in metadata
    assert "tirx-harness-workspace" not in metadata
    assert "file://" not in metadata
    assert not any(
        line.startswith("Requires-Dist: tirx-harness") for line in metadata.splitlines()
    )

    install_root = tmp_path / "installed"
    installed = subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-deps",
            "--target",
            str(install_root),
            str(wheels[0]),
        ],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert installed.returncode == 0, installed.stdout + installed.stderr
    import_env = env.copy()
    import_env["PYTHONPATH"] = os.pathsep.join(
        [str(install_root), import_env.get("PYTHONPATH", "")]
    )
    imported = subprocess.run(
        [
            sys.executable,
            "-c",
            "import pathlib, tirx_harness.numsim as n; "
            "print(pathlib.Path(n.__file__).resolve()); print(n.Engine.__name__)",
        ],
        cwd=tmp_path,
        env=import_env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert imported.returncode == 0, imported.stdout + imported.stderr
    lines = imported.stdout.splitlines()
    assert Path(lines[0]).is_relative_to(install_root)
    assert lines[1] == "Engine"

    smoke_source = "\n".join(
        (
            "import pathlib",
            "import numpy as np",
            "from tvm.script import tirx as T",
            "from tirx_harness import numsim",
            "@T.prim_func",
            "def wheel_add_one(source: T.Buffer((32,), 'float32'), "
            "output: T.Buffer((32,), 'float32')):",
            "    T.device_entry()",
            "    _warp = T.warp_id([1])",
            "    lane = T.lane_id([32])",
            "    output[lane] = source[lane] + T.float32(1)",
            "source = np.arange(32, dtype=np.float32)",
            "module = numsim.transpile(wheel_add_one, cache_dir=pathlib.Path('cache'))",
            "metadata = module.load().metadata()",
            "assert metadata['engine_hash'] == metadata['build_identity']['engine_hash']",
            "assert metadata['build_identity'] == module.artifact.manifest['build_identity']",
            "result = numsim.Engine(max_workers=1).run(",
            "    module, {'source': source, 'output': np.zeros_like(source)}",
            ")",
            "np.testing.assert_array_equal(result.outputs['output'], source + 1)",
            "print('wheel-numsim-smoke-ok')",
        )
    )
    smoke_script = tmp_path / "wheel_numsim_smoke.py"
    smoke_script.write_text(smoke_source, encoding="utf-8")
    smoke = subprocess.run(
        [sys.executable, str(smoke_script)],
        cwd=tmp_path,
        env={
            **env,
            "PYTHONPATH": os.pathsep.join(
                (str(install_root), str(tvm_ffi_root), str(tvm_python_root))
            ),
            "PYTHONNOUSERSITE": "1",
            "CARGO_TARGET_DIR": str(tmp_path / "cargo-target"),
        },
        capture_output=True,
        text=True,
        check=False,
    )
    assert smoke.returncode == 0, smoke.stdout + smoke.stderr
    assert smoke.stdout.splitlines()[-1] == "wheel-numsim-smoke-ok"
