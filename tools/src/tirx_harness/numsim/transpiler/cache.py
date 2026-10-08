"""Content-addressed artifact cache helpers."""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import tomllib
from functools import cache
from pathlib import Path

PYO3_VERSION = "0.29.0"


def default_cache_root() -> Path:
    explicit = os.environ.get("NUMSIM_CACHE_DIR")
    if explicit:
        return Path(explicit).expanduser().resolve()
    xdg = os.environ.get("XDG_CACHE_HOME")
    base = Path(xdg).expanduser() if xdg else Path.home() / ".cache"
    return (base / "tirx-numsim").resolve()


def default_build_cache_root() -> Path:
    """Return the stable native-build cache, independent of artifact destinations."""

    explicit = os.environ.get("NUMSIM_BUILD_CACHE_DIR")
    if explicit:
        return Path(explicit).expanduser().resolve()
    xdg = os.environ.get("XDG_CACHE_HOME")
    base = Path(xdg).expanduser() if xdg else Path.home() / ".cache"
    return (base / "tirx-numsim" / "build").resolve()


def rust_tool(name: str) -> str:
    explicit = os.environ.get(f"NUMSIM_{name.upper()}")
    home_candidate = Path.home() / ".cargo" / "bin" / name
    candidate = explicit if explicit else str(home_candidate) if home_candidate.is_file() else name
    resolved = shutil.which(str(Path(candidate).expanduser()))
    if resolved is None:
        raise FileNotFoundError(f"cannot resolve NumSim Rust tool {name!r} from {candidate!r}")
    # Preserve rustup proxy names such as ``~/.cargo/bin/rustc``. Resolving the
    # symlink to the ``rustup`` binary changes argv[0] and therefore tool mode.
    return str(Path(resolved).absolute())


@cache
def _tool_version(tool: str) -> str:
    completed = subprocess.run(
        [tool, "--version", "--verbose"], check=True, capture_output=True, text=True
    )
    return completed.stdout.strip()


def rustc_version(tool: str | None = None) -> str:
    return _tool_version(rust_tool("rustc") if tool is None else tool)


def cargo_version(tool: str | None = None) -> str:
    return _tool_version(rust_tool("cargo") if tool is None else tool)


def rustc_host_target(tool: str | None = None) -> str:
    for line in rustc_version(tool).splitlines():
        if line.startswith("host: "):
            return line.removeprefix("host: ").strip()
    raise RuntimeError("rustc --version --verbose did not report a host target")


def hash_tree(root: Path) -> str:
    digest = hashlib.sha256()
    for directory, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(name for name in dirnames if name != "target")
        directory_path = Path(directory)
        for filename in sorted(filenames):
            path = directory_path / filename
            if not path.is_file():
                continue
            digest.update(str(path.relative_to(root)).encode())
            digest.update(b"\0")
            digest.update(path.read_bytes())
            digest.update(b"\0")
    return digest.hexdigest()


def engine_build_input_paths(root: Path) -> tuple[Path, ...]:
    """Return files that can affect a generated artifact's engine build.

    Cargo ignores a path dependency's documentation, tests, and package-local
    lockfile when the dependency is built by the generated artifact.  Hash the
    package manifests, library source trees, local path dependencies, build
    scripts, and the generated-artifact lockfile instead of the entire staged
    engine checkout.
    """

    root = root.resolve()
    manifests: list[Path] = [root / "Cargo.toml"]
    visited: set[Path] = set()
    inputs: set[Path] = set()

    while manifests:
        manifest = manifests.pop().resolve()
        if manifest in visited:
            continue
        if not manifest.is_file():
            raise FileNotFoundError(f"NumSim Rust package manifest is missing: {manifest}")
        visited.add(manifest)
        inputs.add(manifest)

        package_root = manifest.parent
        data = tomllib.loads(manifest.read_text())
        source_root = package_root / "src"
        if source_root.is_dir():
            inputs.update(path for path in source_root.rglob("*") if path.is_file())

        package = data.get("package", {})
        build_script = package.get("build")
        if build_script is not False:
            candidate = package_root / (str(build_script) if build_script else "build.rs")
            if candidate.is_file():
                inputs.add(candidate)

        dependency_tables = [
            data.get("dependencies", {}),
            data.get("build-dependencies", {}),
        ]
        for target in data.get("target", {}).values():
            dependency_tables.extend(
                (target.get("dependencies", {}), target.get("build-dependencies", {}))
            )
        for dependencies in dependency_tables:
            for dependency in dependencies.values():
                if not isinstance(dependency, dict) or "path" not in dependency:
                    continue
                dependency_manifest = (
                    package_root / str(dependency["path"]) / "Cargo.toml"
                ).resolve()
                try:
                    dependency_manifest.relative_to(root)
                except ValueError as error:
                    raise ValueError(
                        f"NumSim engine build input escapes the engine root: {dependency_manifest}"
                    ) from error
                manifests.append(dependency_manifest)

    artifact_lock = root / "artifact-template" / "Cargo.lock"
    if not artifact_lock.is_file():
        raise FileNotFoundError(f"NumSim generated-artifact lockfile is missing: {artifact_lock}")
    inputs.add(artifact_lock)
    return tuple(sorted(inputs))


def hash_engine_build_inputs(root: Path) -> str:
    """Hash only engine files consumed by generated-artifact compilation."""

    root = root.resolve()
    digest = hashlib.sha256()
    for path in engine_build_input_paths(root):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()
