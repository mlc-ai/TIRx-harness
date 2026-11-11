"""Package the full harness environment and build the Rust/TVM frontend."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import tomllib
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext
from setuptools.command.build_py import build_py

ROOT = Path(__file__).resolve().parent
FRONTEND = ROOT / "tools" / "frontend-rs"
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


class RustBuildExt(build_ext):
    def copy_extensions_to_source(self) -> None:
        super().copy_extensions_to_source()
        build_py = self.get_finalized_command("build_py")
        for ext in self.extensions:
            inplace, regular = self._get_inplace_equivalent(build_py, ext)
            source, destination = Path(regular).parent, Path(inplace).parent
            shutil.copy2(source / "_tvm_rust_ext.identity", destination)

    def build_extension(self, ext: Extension) -> None:
        Path(self.build_temp).mkdir(parents=True, exist_ok=True)
        # A fresh build directory prevents Cargo reusing an expired isolated FFI path.
        with TemporaryDirectory(dir=Path(self.build_temp).resolve()) as temporary:
            staging = Path(temporary)
            shutil.copytree(
                FRONTEND,
                staging,
                dirs_exist_ok=True,
                ignore=shutil.ignore_patterns("target"),
            )
            shutil.copy2(ROOT / "tools/src/tirx_harness/numsim/dtype_registry.json", staging)
            env = os.environ.copy()
            env["CARGO_TARGET_DIR"] = str(staging / "target")
            # tvm-ffi-config and tvm-tirx-config locate the libraries the crates link.
            scripts = Path(sys.executable).parent
            env["PATH"] = os.pathsep.join((str(scripts), env.get("PATH", "")))
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
            lock = tomllib.loads((staging / "Cargo.lock").read_text())
            binding = next(p for p in lock["package"] if p["name"] == "tvm-tirx")
            bindings = binding["version"]
            output.with_name("_tvm_rust_ext.identity").write_text(
                f"{version}:tvm-tirx-{bindings}\n"
            )


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


setup(
    install_requires=list(dependencies("harness")),
    # This TVM FFI library has no CPython API dependency.
    ext_modules=[Extension("tirx_harness.numsim._tvm_rust_ext", sources=[], py_limited_api=True)],
    cmdclass={"build_ext": RustBuildExt, "build_py": SkillsBuildPy},
    options={"bdist_wheel": {"py_limited_api": "cp312"}},
)
