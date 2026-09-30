"""Package the full harness environment and build the Rust/TVM frontend."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import tomllib
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext
from setuptools.command.build_py import build_py
from setuptools.command.sdist import sdist

ROOT = Path(__file__).resolve().parent
FRONTEND = ROOT / "tirx_harness" / "frontend-rs"
SKILLS = ROOT / "skills"
DEPENDENCY_GROUPS = tomllib.loads((ROOT / "pyproject.toml").read_text())["dependency-groups"]


def dependencies(group: str):
    for entry in DEPENDENCY_GROUPS[group]:
        if isinstance(entry, str):
            yield entry
        else:
            yield from dependencies(entry["include-group"])


def git(*args: str, cwd: Path = ROOT) -> str:
    return subprocess.check_output(["git", *args], cwd=cwd, text=True).strip()


def copy_thirdparty(destination: Path) -> str:
    source = FRONTEND / "thirdparty" / "tvm-rust-ext"
    if not source.is_dir():
        source = ROOT / "thirdparty" / "tvm-rust-ext"
    if not (source / "Cargo.toml").is_file():
        raise RuntimeError("Initialize thirdparty/tvm-rust-ext before building tirx-harness")
    if (source / ".git").exists():
        revision = git("rev-parse", "HEAD", cwd=source)
        if git("status", "--porcelain", cwd=source):
            raise RuntimeError("tvm-rust-ext submodule must be clean before building")
    else:
        revision = (source / "REVISION").read_text().strip()
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise RuntimeError("tvm-rust-ext revision is malformed")
    if (ROOT / ".git").exists() and revision != git("rev-parse", ":thirdparty/tvm-rust-ext"):
        raise RuntimeError("tvm-rust-ext sources do not match the pinned submodule")
    destination.mkdir(parents=True, exist_ok=True)
    for name in ("Cargo.toml", "build.rs", "LICENSE", "NOTICE"):
        shutil.copy2(source / name, destination / name)
    shutil.copytree(source / "src", destination / "src", dirs_exist_ok=True)
    (destination / "REVISION").write_text(revision + "\n")
    return revision


class RustBuildExt(build_ext):
    def copy_extensions_to_source(self) -> None:
        super().copy_extensions_to_source()
        build_py = self.get_finalized_command("build_py")
        for ext in self.extensions:
            inplace, regular = self._get_inplace_equivalent(build_py, ext)
            source, destination = Path(regular).parent, Path(inplace).parent
            shutil.copy2(source / "_tvm_rust_ext.identity", destination)
            shutil.copytree(
                source / "_thirdparty_licenses",
                destination / "_thirdparty_licenses",
                dirs_exist_ok=True,
            )

    def build_extension(self, ext: Extension) -> None:
        Path(self.build_temp).mkdir(parents=True, exist_ok=True)
        # A fresh build directory prevents Cargo reusing an expired isolated FFI path.
        with TemporaryDirectory(dir=Path(self.build_temp).resolve()) as temporary:
            staging = Path(temporary)
            shutil.copytree(
                FRONTEND,
                staging,
                dirs_exist_ok=True,
                ignore=shutil.ignore_patterns("target", "thirdparty"),
            )
            shutil.copy2(ROOT / "tirx_harness/src/tirx_harness/numsim/dtype_registry.json", staging)
            bindings = staging / "thirdparty" / "tvm-rust-ext"
            revision = copy_thirdparty(bindings)
            env = os.environ.copy()
            env["CARGO_TARGET_DIR"] = str(staging / "target")
            env.setdefault("TVM_PYTHON", sys.executable)
            ffi_config = Path(sys.executable).parent / "tvm-ffi-config"
            if ffi_config.is_file():
                env.setdefault("TVM_FFI_CONFIG", str(ffi_config))
            env["PATH"] = os.pathsep.join((str(ffi_config.parent), env.get("PATH", "")))
            subprocess.run(
                [
                    env.get("CARGO", "cargo"),
                    "build",
                    "--release",
                    "--locked",
                    "--manifest-path",
                    str(staging / "Cargo.toml"),
                ],
                env=env,
                check=True,
            )
            library = {
                "win32": "numsim_tirx_frontend.dll",
                "darwin": "libnumsim_tirx_frontend.dylib",
            }.get(sys.platform, "libnumsim_tirx_frontend.so")
            output = Path(self.get_ext_fullpath(ext.name))
            output.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(staging / "target" / "release" / library, output)
            version = tomllib.loads((staging / "Cargo.toml").read_text())["package"]["version"]
            output.with_name("_tvm_rust_ext.identity").write_text(f"{version}:{revision}\n")
            licenses = output.parent / "_thirdparty_licenses" / "tvm-rust-ext"
            licenses.mkdir(parents=True, exist_ok=True)
            for name in ("LICENSE", "NOTICE"):
                shutil.copy2(bindings / name, licenses / name)


def skill_files(skills_dir: Path) -> list[Path]:
    """Return the source files of the skills in ``skills_dir``, excluding fetched references."""
    skill_dirs = sorted(path.parent for path in skills_dir.glob("*/SKILL.md"))
    if (skills_dir.parent / ".git").exists():
        # A checkout may hold fetched references; only tracked files are sources.
        listed = git("ls-files", "-z", "--", *(skill.name for skill in skill_dirs), cwd=skills_dir)
        return [
            path for name in listed.split("\0") if name and (path := skills_dir / name).is_file()
        ]
    # Source distributions contain only tracked files.
    return [
        path
        for skill in skill_dirs
        for path in sorted(skill.rglob("*"))
        if path.is_file() and "__pycache__" not in path.parts
    ]


class SkillsBuildPy(build_py):
    def run(self) -> None:
        super().run()
        if self.editable_mode:
            return
        destination = Path(self.build_lib) / "tirx_harness" / "_skills"
        shutil.rmtree(destination, ignore_errors=True)
        for source in skill_files(SKILLS):
            target = destination / source.relative_to(SKILLS)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)


class RustSdist(sdist):
    def make_release_tree(self, base_dir: str, files: list[str]) -> None:
        super().make_release_tree(base_dir, files)
        copy_thirdparty(Path(base_dir) / "tirx_harness/frontend-rs/thirdparty/tvm-rust-ext")


setup(
    install_requires=list(dependencies("harness")),
    # This TVM FFI library has no CPython API dependency.
    ext_modules=[Extension("tirx_harness.numsim._tvm_rust_ext", sources=[], py_limited_api=True)],
    cmdclass={"build_ext": RustBuildExt, "build_py": SkillsBuildPy, "sdist": RustSdist},
    options={"bdist_wheel": {"py_limited_api": "cp312"}},
)
