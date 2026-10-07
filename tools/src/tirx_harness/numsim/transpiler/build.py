"""Build and load hash-named PyO3 artifacts emitted by NumSim."""

from __future__ import annotations

import fcntl
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import weakref
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from types import ModuleType
from typing import Any, Literal

import numpy as np

from ..abi import abi_metadata
from ..errors import NumSimBuildError
from .cache import (
    PYO3_VERSION,
    cargo_version,
    default_build_cache_root,
    default_cache_root,
    hash_engine_build_inputs,
    rust_tool,
    rustc_host_target,
    rustc_version,
)
from .frontend import ModuleSpec, module_spec_from_manifest
from .semantic_ir import semantic_ir_json

_LOADED: dict[str, ModuleType] = {}
_PREPARED_DEPENDENCIES: dict[str, RustDependencies] = {}
_GENERATED_SOURCE_IDENTITY_LOCK = threading.Lock()
_GENERATED_SOURCE_IDENTITIES: weakref.WeakKeyDictionary[Any, str] = weakref.WeakKeyDictionary()
_CARGO_PACKAGE_NAME = "numsim-generated-artifact"
_GENERATED_BACKEND = "direct-rustc-v1"
_GENERATED_ARTIFACT_RELEASE_OPT_LEVEL = 3
_GENERATED_OPT_LEVEL = _GENERATED_ARTIFACT_RELEASE_OPT_LEVEL
_GENERATED_CODEGEN_UNITS = 16
_GENERATED_RUSTC_THREADS = 8
_ANALYSIS_GENERATED_CODEGEN_UNITS = 32
# Racecheck retains its execution-tuned cross-unit optimization boundary.
_RACECHECK_GENERATED_CODEGEN_UNITS = 6
_ANALYSIS_GENERATED_RUSTC_THREADS = 8
_GENERATED_ARTIFACT_O0_CODEGEN_UNITS = 32
# Generated kernels call precompiled semantic engine boundaries instead of
# depending on cross-crate inlining. Spell this explicitly for every artifact,
# including Racecheck: ThinLTO would ingest and re-optimize the engine bitcode
# once per kernel, defeating the v2 ABI boundary.
_GENERATED_ARTIFACT_RUSTC_LTO: str | None = "off"
_GENERATED_ARTIFACT_TARGET_CPU = "native"
_RACECHECK_GENERATED_TARGET_CPU = _GENERATED_ARTIFACT_TARGET_CPU
_GENERATED_OPT_LEVEL_ENV = "NUMSIM_GENERATED_OPT_LEVEL"
_GENERATED_STRIP_ENV = "NUMSIM_GENERATED_STRIP"
_GENERATED_CACHE_KIND = "generated"
_MODULE_PLACEHOLDER = "__NUMSIM_MODULE__"
_CACHE_KEY_PLACEHOLDER = "__NUMSIM_CACHE_KEY__"
_CACHE_KEY_HEX_CHARS = hashlib.sha256().digest_size * 2
_MODULE_KEY_HEX_CHARS = 24
_ENGINE_HASH_ENV = "NUMSIM_ARTIFACT_ENGINE_HASH"
_BUILD_IDENTITY_ENV = "NUMSIM_ARTIFACT_BUILD_IDENTITY_JSON"
_BUILD_ENVIRONMENT_KEYS = (
    "AR",
    "CC",
    "CFLAGS",
    "CARGO_HOME",
    "CXX",
    "CXXFLAGS",
    "LDFLAGS",
    "MACOSX_DEPLOYMENT_TARGET",
    "PATH",
    "PKG_CONFIG_PATH",
    "RUSTUP_HOME",
    "RUSTUP_TOOLCHAIN",
)


def _profile_enabled() -> bool:
    return os.environ.get("NUMSIM_PROFILE") == "1"


def _generated_artifact_release_opt_level(
    *, default: int = _GENERATED_ARTIFACT_RELEASE_OPT_LEVEL
) -> int:
    if default not in {0, 1, 3}:
        raise NumSimBuildError("generated artifact default opt level must be 0, 1, or 3")
    raw = os.environ.get(_GENERATED_OPT_LEVEL_ENV)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError as error:
        raise NumSimBuildError(f"{_GENERATED_OPT_LEVEL_ENV} must be 0, 1, or 3") from error
    if value not in {0, 1, 3}:
        raise NumSimBuildError(f"{_GENERATED_OPT_LEVEL_ENV} must be 0, 1, or 3")
    return value


def _generated_artifact_strip() -> str | None:
    raw = os.environ.get(_GENERATED_STRIP_ENV)
    if raw in {None, ""}:
        return "debuginfo"
    if raw == "none":
        return None
    if raw not in {"debuginfo", "symbols"}:
        raise NumSimBuildError(f"{_GENERATED_STRIP_ENV} must be one of: none, debuginfo, symbols")
    return raw


@dataclass(frozen=True)
class GeneratedBuildConfig:
    profile_enabled: bool
    release_opt_level: int
    rustc_lto: str | None
    target_cpu: str | None
    codegen_units: int
    rustc_threads: int
    strip: str | None


def _generated_build_config(
    *,
    default_opt_level: int = _GENERATED_ARTIFACT_RELEASE_OPT_LEVEL,
    analysis_capable: bool = False,
    analysis_checker: Literal["synccheck", "racecheck"] | None = None,
) -> GeneratedBuildConfig:
    if analysis_checker is not None and not analysis_capable:
        raise ValueError("an analysis checker requires analysis-capable code generation")
    release_opt_level = _generated_artifact_release_opt_level(default=default_opt_level)
    optimized_analysis = analysis_capable and release_opt_level == 3
    return GeneratedBuildConfig(
        profile_enabled=_profile_enabled(),
        release_opt_level=release_opt_level,
        rustc_lto=_GENERATED_ARTIFACT_RUSTC_LTO,
        target_cpu=(
            _RACECHECK_GENERATED_TARGET_CPU
            if analysis_checker == "racecheck" and release_opt_level == 3
            else (_GENERATED_ARTIFACT_TARGET_CPU if release_opt_level == 3 else None)
        ),
        codegen_units=(
            _GENERATED_ARTIFACT_O0_CODEGEN_UNITS
            if release_opt_level == 0
            else (
                (
                    _RACECHECK_GENERATED_CODEGEN_UNITS
                    if analysis_checker == "racecheck"
                    else _ANALYSIS_GENERATED_CODEGEN_UNITS
                )
                if optimized_analysis
                else _GENERATED_CODEGEN_UNITS
            )
        ),
        rustc_threads=(
            _ANALYSIS_GENERATED_RUSTC_THREADS if optimized_analysis else _GENERATED_RUSTC_THREADS
        ),
        strip=_generated_artifact_strip(),
    )


@dataclass(frozen=True)
class Artifact:
    key: str
    module_name: str
    directory: Path
    library_path: Path
    source: str
    manifest: dict[str, Any]
    spec: ModuleSpec | None = None

    def load(self) -> ModuleType:
        loaded = _LOADED.get(self.key)
        if loaded is not None:
            self.validate_native_metadata(_read_native_metadata(loaded, self.library_path))
            return loaded
        spec = importlib.util.spec_from_file_location(self.module_name, self.library_path)
        if spec is None or spec.loader is None:
            raise NumSimBuildError(f"Cannot create import spec for {self.library_path}")
        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
        except Exception as error:
            raise NumSimBuildError(
                f"Cannot load NumSim artifact extension {self.library_path}: "
                f"{type(error).__name__}: {error}"
            ) from error
        self.validate_native_metadata(_read_native_metadata(module, self.library_path))
        _LOADED[self.key] = module
        return module

    def validate_native_metadata(self, native_metadata: Any) -> None:
        """Validate one native metadata snapshot against the manifest and Python ABI."""

        if not isinstance(native_metadata, dict):
            raise NumSimBuildError("Loaded artifact metadata must be a dictionary")
        manifest_identity = self.manifest.get("build_identity")
        if not isinstance(manifest_identity, dict):
            raise NumSimBuildError("Artifact manifest build_identity must be a dictionary")
        manifest_engine_hash = self.manifest.get("engine_hash")
        identity_engine_hash = manifest_identity.get("engine_hash")
        if manifest_engine_hash != identity_engine_hash:
            raise NumSimBuildError(
                "Artifact manifest engine_hash disagrees with "
                "build_identity.engine_hash: "
                f"engine_hash={manifest_engine_hash!r}, "
                f"build_identity={identity_engine_hash!r}"
            )

        native_identity = native_metadata.get("build_identity")
        if not isinstance(native_identity, dict):
            raise NumSimBuildError("Loaded artifact build_identity must be a dictionary")
        native_engine_hash = native_metadata.get("engine_hash")
        native_identity_engine_hash = native_identity.get("engine_hash")
        if native_engine_hash != native_identity_engine_hash:
            raise NumSimBuildError(
                "Loaded artifact engine_hash disagrees with build_identity.engine_hash: "
                f"engine_hash={native_engine_hash!r}, "
                f"build_identity={native_identity_engine_hash!r}"
            )

        expected_fields = {
            "cache_key": self.key,
            "engine_hash": manifest_engine_hash,
        }
        for name, expected in expected_fields.items():
            actual = native_metadata.get(name)
            if actual != expected:
                raise NumSimBuildError(
                    f"Loaded artifact {name} does not match its manifest: "
                    f"artifact={actual!r}, manifest={expected!r}"
                )
        identity_mismatches = _build_identity_mismatches(native_identity, manifest_identity)
        if identity_mismatches:
            raise NumSimBuildError(
                "Loaded artifact build identity does not match its manifest: "
                + "; ".join(identity_mismatches)
            )
        for name, expected in abi_metadata().items():
            actual = native_metadata.get(name)
            if actual != expected:
                raise NumSimBuildError(
                    f"Loaded artifact {name} does not match Python ABI: "
                    f"artifact={actual!r}, python={expected!r}"
                )


@dataclass(frozen=True)
class RustDependencies:
    """Prebuilt Cargo dependencies consumed by parallel generated rustc calls."""

    key: str
    directory: Path
    engine_rlib: Path
    pyo3_rlib: Path


@dataclass(frozen=True)
class PreparedArtifact:
    cache_kind: str
    artifact_kind: str
    codegen_fingerprint: str | None
    generated_source_identity: str | None
    engine_root: Path
    root: Path
    build_root: Path
    cargo_path: str
    rustc_path: str
    cargo_identity: str
    rustc_identity: str
    rust_target: str
    engine_hash: str
    build_config: GeneratedBuildConfig
    generated_profile: dict[str, Any]
    build_identity: dict[str, Any]
    target_dir: Path
    key: str
    module_name: str
    lib_target: str
    artifact_dir: Path
    shared_artifact_dir: Path


def _read_native_metadata(module: ModuleType, library_path: Path) -> Any:
    try:
        return module.metadata()
    except Exception as error:
        raise NumSimBuildError(
            f"Loaded NumSim artifact {library_path} rejected its metadata: "
            f"{type(error).__name__}: {error}"
        ) from error


def _engine_root() -> Path:
    return Path(__file__).resolve().parents[1] / "engine-rs"


def _build_cache_root() -> Path:
    return default_build_cache_root()


def _dependency_identity(build_identity: Mapping[str, Any]) -> dict[str, Any]:
    identity = {
        name: value
        for name, value in build_identity.items()
        if name not in {"generated_backend", "generated_profile"}
    }
    generated_profile = build_identity.get("generated_profile")
    if isinstance(generated_profile, Mapping) and generated_profile.get("lto") is not None:
        identity["generated_lto"] = generated_profile["lto"]
    if isinstance(generated_profile, Mapping) and generated_profile.get("target_cpu") is not None:
        identity["generated_target_cpu"] = generated_profile["target_cpu"]
    return identity


def _dependency_key(build_identity: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        _dependency_identity(build_identity), sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _cargo_target_dir(root: Path, build_identity: Mapping[str, Any]) -> Path:
    return root / "cargo-target" / _dependency_key(build_identity)


def _artifact_lock_path(engine_root: Path) -> Path:
    return engine_root / "artifact-template" / "Cargo.lock"


def _ensure_cache_root(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    metadata = root.stat()
    if metadata.st_uid != os.getuid():
        raise NumSimBuildError(f"NumSim cache root is not owned by the current user: {root}")
    if metadata.st_mode & 0o022:
        raise NumSimBuildError(f"NumSim cache root must not be group- or world-writable: {root}")


def _engine_snapshot_dir(root: Path, engine_hash: str) -> Path:
    return root / "engines" / engine_hash


def _stage_engine(engine_root: Path, root: Path, expected_hash: str) -> Path:
    staged = _engine_snapshot_dir(root, expected_hash)
    locks = root / "locks"
    locks.mkdir(parents=True, exist_ok=True)
    with (locks / f"engine-{expected_hash}.lock").open("a+") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        if staged.is_dir():
            actual_hash = hash_engine_build_inputs(staged)
            if actual_hash != expected_hash:
                raise NumSimBuildError(
                    "NumSim cached Rust engine snapshot does not match its content hash: "
                    f"expected={expected_hash}, actual={actual_hash}"
                )
            return staged
        if staged.exists():
            raise NumSimBuildError(f"NumSim engine snapshot path is not a directory: {staged}")

        staged.parent.mkdir(parents=True, exist_ok=True)
        temp_root = root / "tmp"
        temp_root.mkdir(parents=True, exist_ok=True)
        snapshot_temp = Path(tempfile.mkdtemp(prefix=f"engine-{expected_hash}-", dir=temp_root))
        candidate = snapshot_temp / "numsim-engine"
        try:
            shutil.copytree(
                engine_root,
                candidate,
                ignore=shutil.ignore_patterns("target", "__pycache__", "*.pyc"),
            )
            actual_hash = hash_engine_build_inputs(candidate)
            if actual_hash != expected_hash:
                raise NumSimBuildError(
                    "NumSim Rust engine changed while its build snapshot was being created; retry"
                )
            os.replace(candidate, staged)
        finally:
            shutil.rmtree(snapshot_temp, ignore_errors=True)
    return staged


def _cargo_lib_target(key: str) -> str:
    return f"numsim_artifact_{key[:24]}"


def _cargo_configuration(root: Path) -> dict[str, str]:
    candidates: set[Path] = set()
    cargo_home = Path(os.environ.get("CARGO_HOME", Path.home() / ".cargo")).expanduser()
    candidates.update((cargo_home / "config", cargo_home / "config.toml"))
    for parent in (root, *root.parents):
        candidates.update((parent / ".cargo" / "config", parent / ".cargo" / "config.toml"))
    result: dict[str, str] = {}
    for path in sorted(candidates):
        if path.is_file():
            result[str(path.resolve())] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def _build_input_environment(root: Path) -> dict[str, Any]:
    return {
        "environment": {
            name: os.environ[name] for name in _BUILD_ENVIRONMENT_KEYS if name in os.environ
        },
        "cargo_configuration": _cargo_configuration(root),
    }


def _render_generated_source(source_template: str, key: str) -> tuple[str, str]:
    if len(key) != _CACHE_KEY_HEX_CHARS:
        raise NumSimBuildError("generated artifact key is not a SHA-256 hex digest")
    module_name = f"_numsim_{key[:_MODULE_KEY_HEX_CHARS]}"
    source = source_template.replace(_MODULE_PLACEHOLDER, module_name).replace(
        _CACHE_KEY_PLACEHOLDER, key
    )
    return module_name, source


def _generated_profile(
    source_template: str,
    *,
    build_config: GeneratedBuildConfig | None = None,
    default_opt_level: int = _GENERATED_ARTIFACT_RELEASE_OPT_LEVEL,
) -> dict[str, Any]:
    del source_template
    config = (
        _generated_build_config(default_opt_level=default_opt_level)
        if build_config is None
        else build_config
    )
    profile: dict[str, Any] = {
        "opt_level": config.release_opt_level,
        "codegen_units": config.codegen_units,
        "rustc_threads": config.rustc_threads,
    }
    if config.rustc_lto is not None:
        profile["lto"] = config.rustc_lto
    if config.target_cpu is not None:
        profile["target_cpu"] = config.target_cpu
    if config.strip != "debuginfo":
        profile["strip"] = config.strip
    return profile


def _build_identity(
    engine_hash: str,
    *,
    cache_root: Path,
    generated_profile: Mapping[str, Any],
    analysis_capable: bool = False,
    analysis_checker: Literal["synccheck", "racecheck"] | None = None,
    cargo_path: str,
    rustc_path: str,
    cargo_identity: str,
    rustc_identity: str,
    rust_target: str,
) -> dict[str, Any]:
    identity = {
        "engine_hash": engine_hash,
        **abi_metadata(),
        "pyo3": PYO3_VERSION,
        "python_cache_tag": sys.implementation.cache_tag,
        "python_executable": str(Path(sys.executable).resolve()),
        "python_version": sys.version,
        "numpy_version": np.__version__,
        "profile": _profile_enabled(),
        "cargo": cargo_identity,
        "cargo_path": cargo_path,
        "rustc": rustc_identity,
        "rustc_path": rustc_path,
        "rust_target": rust_target,
        "generated_backend": _GENERATED_BACKEND,
        "generated_profile": dict(generated_profile),
        "build_inputs": _build_input_environment(cache_root),
    }
    identity["artifact_kind"] = (
        analysis_checker
        if analysis_checker is not None
        else ("synccheck" if analysis_capable else "numsim")
    )
    return identity


def _canonical_build_identity_json(build_identity: Mapping[str, Any]) -> str:
    return json.dumps(build_identity, sort_keys=True, separators=(",", ":"))


def _identity_path(parent: str, key: str) -> str:
    return f"{parent}.{key}" if key.isidentifier() else f"{parent}[{key!r}]"


def _build_identity_mismatches(
    cached: Any,
    current: Any,
    *,
    path: str = "build_identity",
) -> list[str]:
    """Return deterministic field-level differences between two JSON identities."""

    if isinstance(cached, Mapping) and isinstance(current, Mapping):
        mismatches: list[str] = []
        for key in sorted(set(cached) | set(current)):
            child_path = _identity_path(path, str(key))
            if key not in cached:
                mismatches.append(
                    f"{child_path} missing from cache manifest: current={current[key]!r}"
                )
            elif key not in current:
                mismatches.append(
                    f"{child_path} is unexpected in cache manifest: cached={cached[key]!r}"
                )
            else:
                mismatches.extend(
                    _build_identity_mismatches(cached[key], current[key], path=child_path)
                )
        return mismatches
    if isinstance(cached, list) and isinstance(current, list):
        mismatches = []
        if len(cached) != len(current):
            mismatches.append(
                f"{path} length mismatch: cached={len(cached)!r}, current={len(current)!r}"
            )
        for index, (cached_item, current_item) in enumerate(zip(cached, current, strict=False)):
            mismatches.extend(
                _build_identity_mismatches(
                    cached_item,
                    current_item,
                    path=f"{path}[{index}]",
                )
            )
        return mismatches
    if type(cached) is not type(current) or cached != current:
        return [f"{path} mismatch: cached={cached!r}, current={current!r}"]
    return []


def _compiler_identity() -> dict[str, Any]:
    try:
        import tvm
        import tvm_ffi
        from tvm.base import _LOADED_LIBS
    except ImportError:
        return {"tvm": None}
    libraries = {Path(tvm_ffi.LIB._name).resolve()}
    libraries.update(Path(library._name).resolve() for library in _LOADED_LIBS.values())
    library_identity: list[dict[str, Any]] = []
    for path in sorted(libraries):
        try:
            metadata = path.stat()
        except OSError:
            library_identity.append({"path": str(path), "missing": True})
            continue
        library_identity.append(
            {"path": str(path), "size": metadata.st_size, "mtime_ns": metadata.st_mtime_ns}
        )
    return {
        "tvm_version": getattr(tvm, "__version__", None),
        "tvm_package": str(Path(tvm.__file__).resolve()),
        "tvm_ffi_version": getattr(tvm_ffi, "__version__", None),
        "tvm_ffi_package": str(Path(tvm_ffi.__file__).resolve()),
        "libraries": library_identity,
    }


@cache
def _static_codegen_fingerprint() -> str:
    from .compile_cache import codegen_dependency_source_digest

    digest = hashlib.sha256()
    digest.update(json.dumps(_compiler_identity(), sort_keys=True, separators=(",", ":")).encode())
    digest.update(b"\0")
    digest.update(codegen_dependency_source_digest().encode())
    return digest.hexdigest()


def _codegen_fingerprint() -> str:
    from .compile_cache import split_configuration_identity

    digest = hashlib.sha256()
    digest.update(_static_codegen_fingerprint().encode())
    digest.update(b"\0")
    digest.update(
        json.dumps(split_configuration_identity(), sort_keys=True, separators=(",", ":")).encode()
    )
    return digest.hexdigest()


def generated_source_identity(source: Any) -> str:
    from tvm.tirx import PrimFunc

    funcs = tuple(source) if isinstance(source, (list, tuple)) else (source,)
    if not funcs:
        raise ValueError("NumSim cannot transpile an empty kernel sequence")
    if any(not isinstance(func, PrimFunc) for func in funcs):
        # This identity is only a pre-analysis cache probe.  Keep validation in
        # the analyzer so callers which replace the analyzer/build in tests (or
        # other tooling) retain the historical ordering contract.  Object IDs
        # are process-local and the frozen source remains alive in the compiled
        # module, so an opaque probe cannot collide with a valid TIR artifact.
        payload = {
            "opaque_process": os.getpid(),
            "opaque_sources": [
                {"type": f"{type(func).__module__}.{type(func).__qualname__}", "id": id(func)}
                for func in funcs
            ],
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()
    if len(funcs) == 1:
        with _GENERATED_SOURCE_IDENTITY_LOCK:
            cached = _GENERATED_SOURCE_IDENTITIES.get(funcs[0])
        if cached is not None:
            return cached
    payload = {"kernels": [semantic_ir_json(func) for func in funcs]}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    identity = hashlib.sha256(encoded).hexdigest()
    if len(funcs) == 1:
        with _GENERATED_SOURCE_IDENTITY_LOCK:
            _GENERATED_SOURCE_IDENTITIES[funcs[0]] = identity
    return identity


def artifact_kind(
    *, analysis_capable: bool, analysis_checker: Literal["synccheck", "racecheck"] | None
) -> str:
    """What a generated artifact is: the mode both compile caches key on."""

    if analysis_checker is not None:
        return analysis_checker
    return "synccheck" if analysis_capable else "numsim"


def _generated_key_payload(
    build_identity: Mapping[str, Any],
    *,
    codegen_fingerprint: str,
    generated_source_identity: str,
    artifact_kind: str,
) -> bytes:
    payload = {
        "generated_source_identity": generated_source_identity,
        "codegen_fingerprint": codegen_fingerprint,
        "artifact_kind": artifact_kind,
        "build_identity": build_identity,
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()


def prepare_generated_artifact(
    source: Any,
    *,
    cache_dir: str | Path | None = None,
    analysis_capable: bool = False,
    analysis_checker: Literal["synccheck", "racecheck"] | None = None,
    default_opt_level: int = _GENERATED_ARTIFACT_RELEASE_OPT_LEVEL,
) -> PreparedArtifact:
    if analysis_checker not in {None, "synccheck", "racecheck"}:
        raise ValueError(f"unknown native analysis checker: {analysis_checker!r}")
    if analysis_checker is not None and not analysis_capable:
        raise ValueError("a native analysis checker requires analysis-capable code generation")
    source_identity = generated_source_identity(source)
    kind = artifact_kind(analysis_capable=analysis_capable, analysis_checker=analysis_checker)
    engine_root = _engine_root()
    if not (engine_root / "Cargo.toml").is_file():
        raise NumSimBuildError(f"NumSim Rust engine is missing: {engine_root}")
    if not _artifact_lock_path(engine_root).is_file():
        raise NumSimBuildError(
            f"NumSim generated-artifact lockfile is missing: {_artifact_lock_path(engine_root)}"
        )
    root = Path(cache_dir).expanduser().resolve() if cache_dir else default_cache_root()
    _ensure_cache_root(root)
    build_root = _build_cache_root().expanduser().resolve()
    _ensure_cache_root(build_root)
    cargo_path = rust_tool("cargo")
    rustc_path = rust_tool("rustc")
    cargo_identity = cargo_version(cargo_path)
    rustc_identity = rustc_version(rustc_path)
    rust_target = rustc_host_target(rustc_path)
    engine_hash = hash_engine_build_inputs(engine_root)
    build_config = _generated_build_config(
        default_opt_level=default_opt_level,
        analysis_capable=analysis_capable,
        analysis_checker=analysis_checker,
    )
    generated_profile = _generated_profile("", build_config=build_config)
    build_identity = _build_identity(
        engine_hash,
        cache_root=build_root,
        generated_profile=generated_profile,
        analysis_capable=analysis_capable,
        analysis_checker=analysis_checker,
        cargo_path=cargo_path,
        rustc_path=rustc_path,
        cargo_identity=cargo_identity,
        rustc_identity=rustc_identity,
        rust_target=rust_target,
    )
    target_dir = _cargo_target_dir(build_root, build_identity)
    codegen_fingerprint = _codegen_fingerprint()
    key = hashlib.sha256(
        _generated_key_payload(
            build_identity,
            codegen_fingerprint=codegen_fingerprint,
            generated_source_identity=source_identity,
            artifact_kind=kind,
        )
    ).hexdigest()
    return PreparedArtifact(
        cache_kind=_GENERATED_CACHE_KIND,
        artifact_kind=kind,
        codegen_fingerprint=codegen_fingerprint,
        generated_source_identity=source_identity,
        engine_root=engine_root,
        root=root,
        build_root=build_root,
        cargo_path=cargo_path,
        rustc_path=rustc_path,
        cargo_identity=cargo_identity,
        rustc_identity=rustc_identity,
        rust_target=rust_target,
        engine_hash=engine_hash,
        build_config=build_config,
        generated_profile=generated_profile,
        build_identity=build_identity,
        target_dir=target_dir,
        key=key,
        module_name=_render_generated_source("", key)[0],
        lib_target=_cargo_lib_target(key),
        artifact_dir=root / "artifacts" / key,
        shared_artifact_dir=build_root / "artifacts" / key,
    )


def _reject_cached(diagnostics: list[str] | None, reason: str) -> None:
    if diagnostics is not None:
        diagnostics.append(reason)


def _json_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _read_cached(
    directory: Path,
    key: str,
    source: str | None,
    *,
    expected_build_identity: Mapping[str, Any],
    diagnostics: list[str] | None = None,
    expected_cache_kind: str | None = None,
    expected_codegen_fingerprint: str | None = None,
    expected_generated_source_identity: str | None = None,
    reuse_from: Artifact | None = None,
) -> Artifact | None:
    manifest_path = directory / "manifest.json"
    library_path = directory / "module.so"
    if not manifest_path.is_file() or not library_path.is_file():
        _reject_cached(diagnostics, "cache entry is missing manifest.json or module.so")
        return None
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        _reject_cached(
            diagnostics,
            f"cache manifest cannot be read: {type(error).__name__}: {error}",
        )
        return None
    expected_abi = abi_metadata()
    if manifest.get("cache_key") != key:
        _reject_cached(
            diagnostics,
            f"cache_key mismatch: cached={manifest.get('cache_key')!r}, current={key!r}",
        )
        return None
    manifest_identity = manifest.get("build_identity")
    if not isinstance(manifest_identity, dict):
        _reject_cached(diagnostics, "cache manifest field 'build_identity' is not an object")
        return None
    manifest_engine_hash = manifest.get("engine_hash")
    identity_engine_hash = manifest_identity.get("engine_hash")
    identity_rejections = []
    if manifest_engine_hash != identity_engine_hash:
        identity_rejections.append(
            "engine_hash disagrees with build_identity.engine_hash: "
            f"engine_hash={manifest_engine_hash!r}, build_identity={identity_engine_hash!r}"
        )
    identity_rejections.extend(
        _build_identity_mismatches(manifest_identity, expected_build_identity)
    )
    if identity_rejections:
        for reason in identity_rejections:
            _reject_cached(diagnostics, reason)
        return None
    for name, expected in expected_abi.items():
        actual = manifest.get(name)
        if actual != expected:
            _reject_cached(
                diagnostics,
                f"{name} mismatch: cached={actual!r}, current={expected!r}",
            )
            return None
    for name in ("module_name",):
        if not isinstance(manifest.get(name), str):
            _reject_cached(diagnostics, f"cache manifest field {name!r} is not a string")
            return None
    expected_fields = {
        "cache_kind": expected_cache_kind,
        "codegen_fingerprint": expected_codegen_fingerprint,
        "generated_source_identity": expected_generated_source_identity,
    }
    for name, expected in expected_fields.items():
        if expected is not None and manifest.get(name) != expected:
            _reject_cached(
                diagnostics,
                f"{name} mismatch: cached={manifest.get(name)!r}, current={expected!r}",
            )
            return None
    if not isinstance(manifest.get("cache_rebuild_reasons"), list) or not all(
        isinstance(reason, str) for reason in manifest.get("cache_rebuild_reasons", ())
    ):
        _reject_cached(
            diagnostics,
            "cache manifest field 'cache_rebuild_reasons' is not a string list",
        )
        return None
    if source is None:
        source_path = directory / "src" / "lib.rs"
        try:
            source = source_path.read_text()
        except (OSError, UnicodeError) as error:
            _reject_cached(
                diagnostics,
                f"cached generated source cannot be read: {type(error).__name__}: {error}",
            )
            return None
    source_sha256 = hashlib.sha256(source.encode()).hexdigest()
    if manifest.get("source_sha256") != source_sha256:
        _reject_cached(
            diagnostics,
            "source_sha256 mismatch: "
            f"cached={manifest.get('source_sha256')!r}, current={source_sha256!r}",
        )
        return None
    try:
        library_hash = hashlib.sha256(library_path.read_bytes()).hexdigest()
    except OSError as error:
        _reject_cached(
            diagnostics,
            f"cached module cannot be read: {type(error).__name__}: {error}",
        )
        return None
    if manifest.get("library_sha256") != library_hash:
        _reject_cached(
            diagnostics,
            "library_sha256 mismatch: "
            f"cached={manifest.get('library_sha256')!r}, current={library_hash!r}",
        )
        return None
    cached_spec = None
    if expected_cache_kind == _GENERATED_CACHE_KIND:
        if manifest.get("spec_sha256") != _json_sha256(manifest.get("spec")):
            _reject_cached(diagnostics, "spec_sha256 mismatch")
            return None
        try:
            # Reuse metadata restored earlier in this build only after comparing
            # the actual on-disk metadata in full.
            if (
                reuse_from is not None
                and reuse_from.spec is not None
                and manifest.get("spec") == reuse_from.manifest.get("spec")
            ):
                cached_spec = reuse_from.spec
            else:
                cached_spec = module_spec_from_manifest(manifest.get("spec"))
        except (TypeError, ValueError) as error:
            _reject_cached(
                diagnostics,
                f"generated metadata cannot be restored: {type(error).__name__}: {error}",
            )
            return None
    return Artifact(
        key=key,
        module_name=manifest["module_name"],
        directory=directory,
        library_path=library_path,
        source=source,
        manifest=manifest,
        spec=cached_spec,
    )


def _read_prepared_generated(
    directory: Path,
    prepared: PreparedArtifact,
    *,
    diagnostics: list[str] | None = None,
    reuse_from: Artifact | None = None,
) -> Artifact | None:
    return _read_cached(
        directory,
        prepared.key,
        None,
        expected_build_identity=prepared.build_identity,
        diagnostics=diagnostics,
        reuse_from=reuse_from,
        expected_cache_kind=prepared.cache_kind,
        expected_codegen_fingerprint=prepared.codegen_fingerprint,
        expected_generated_source_identity=prepared.generated_source_identity,
    )


def load_cached_generated_artifact(prepared: PreparedArtifact) -> Artifact | None:
    if prepared.cache_kind != _GENERATED_CACHE_KIND:
        raise ValueError("prepared artifact is not a generated-source cache request")
    cache_rejections: list[str] = []
    cached = _read_prepared_generated(prepared.artifact_dir, prepared, diagnostics=cache_rejections)
    if cached is not None:
        return cached

    local_locks = prepared.root / "locks"
    local_locks.mkdir(parents=True, exist_ok=True)
    with (local_locks / f"{prepared.key}.lock").open("a+") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        cached = _read_prepared_generated(
            prepared.artifact_dir, prepared, diagnostics=cache_rejections
        )
        if cached is not None:
            return cached
        shared = _read_prepared_generated(prepared.shared_artifact_dir, prepared)
        if shared is None:
            return None
        _materialize_artifact(
            shared.directory,
            prepared.artifact_dir,
            root=prepared.root,
            cache_rejections=cache_rejections,
        )
    return _read_prepared_generated(prepared.artifact_dir, prepared, reuse_from=shared)


def _cargo_toml(
    engine_root: Path,
    lib_target: str,
    *,
    generated_profile: Mapping[str, Any] | None = None,
    build_config: GeneratedBuildConfig | None = None,
    analysis_capable: bool = False,
    analysis_checker: Literal["synccheck", "racecheck"] | None = None,
) -> str:
    if generated_profile is not None and build_config is not None:
        raise ValueError("supply either generated_profile or build_config, not both")
    if analysis_checker not in {None, "synccheck", "racecheck"}:
        raise ValueError(f"unknown native analysis checker: {analysis_checker!r}")
    if analysis_checker is not None and not analysis_capable:
        raise ValueError("a native analysis checker requires analysis-capable code generation")
    config = (
        _generated_build_config(analysis_capable=analysis_capable)
        if build_config is None
        else build_config
    )
    engine_features = ["python"]
    if analysis_capable:
        # Preserve `analysis` as the compatibility umbrella for Racecheck while
        # keeping its large global-shadow implementation out of Synccheck's
        # dependency metadata and generated specialization.
        engine_features.append("analysis" if analysis_checker == "racecheck" else "analysis-core")
    if config.profile_enabled:
        engine_features.append("profile")
    profile = (
        _generated_profile("", build_config=config)
        if generated_profile is None
        else generated_profile
    )
    strip_line = "" if profile.get("strip") is None else f"strip = {json.dumps(profile['strip'])}\n"
    lto_profile = (
        ""
        if profile.get("lto") is None
        else f"[profile.release]\nlto = {json.dumps(profile['lto'])}\n\n"
    )
    return f"""[package]
name = {json.dumps(_CARGO_PACKAGE_NAME)}
version = "0.0.0"
edition = "2024"
publish = false

[lib]
name = {json.dumps(lib_target)}
crate-type = ["cdylib"]

[dependencies]
numsim-engine = {{ path = {json.dumps(str(engine_root))}, features = {json.dumps(engine_features)} }}
pyo3 = {{ version = "={PYO3_VERSION}", features = ["extension-module"] }}

{lto_profile}[profile.release.package.{json.dumps(_CARGO_PACKAGE_NAME)}]
opt-level = {profile["opt_level"]}
codegen-units = {profile["codegen_units"]}
{strip_line}
"""


def _build_environment(
    target_dir: Path,
    *,
    cargo_path: str,
    rustc_path: str,
    engine_hash: str,
    build_identity: Mapping[str, Any],
) -> dict[str, str]:
    env = os.environ.copy()
    for name in tuple(env):
        if name in {
            "CARGO_ENCODED_RUSTFLAGS",
            "RUSTFLAGS",
            "RUSTC_WRAPPER",
            "RUSTC_WORKSPACE_WRAPPER",
        } or name.startswith(("CARGO_BUILD_", "CARGO_PROFILE_", "CARGO_TARGET_", "PYO3_")):
            env.pop(name, None)
    env["PYO3_PYTHON"] = sys.executable
    env["CARGO_INCREMENTAL"] = "0"
    env["CARGO_TARGET_DIR"] = str(target_dir)
    env["RUSTC"] = rustc_path
    generated_profile = build_identity.get("generated_profile")
    if isinstance(generated_profile, Mapping) and generated_profile.get("target_cpu") is not None:
        env["CARGO_ENCODED_RUSTFLAGS"] = f"-C\x1ftarget-cpu={generated_profile['target_cpu']}"
    env["PATH"] = f"{Path(cargo_path).parent}:{env.get('PATH', '')}"
    env[_ENGINE_HASH_ENV] = engine_hash
    env[_BUILD_IDENTITY_ENV] = _canonical_build_identity_json(build_identity)
    return env


def _dependency_cache_dir(root: Path, key: str) -> Path:
    return root / "dependencies" / key


def _dependency_rlib(directory: Path, crate_name: str) -> Path:
    candidates = sorted(directory.glob(f"lib{crate_name}-*.rlib"))
    if len(candidates) != 1:
        raise NumSimBuildError(
            f"Cargo produced {len(candidates)} {crate_name!r} rlib candidates: {candidates}"
        )
    return candidates[0]


def _read_prepared_dependencies(
    directory: Path,
    key: str,
    *,
    expected_identity: Mapping[str, Any],
    target_dir: Path,
) -> RustDependencies | None:
    memory_key = str(directory)
    prepared = _PREPARED_DEPENDENCIES.get(memory_key)
    if prepared is not None:
        return prepared

    manifest_path = directory / "manifest.json"
    if not manifest_path.is_file():
        return None
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if manifest.get("cache_key") != key:
        return None
    if manifest.get("dependency_identity") != expected_identity:
        return None
    if manifest.get("generated_backend") != _GENERATED_BACKEND:
        return None

    dependency_dir = target_dir / "release" / "deps"
    resolved: dict[str, Path] = {}
    for name in ("engine_rlib", "pyo3_rlib"):
        relative = manifest.get(name)
        expected_hash = manifest.get(f"{name}_sha256")
        if not isinstance(relative, str) or not isinstance(expected_hash, str):
            return None
        path = target_dir / relative
        try:
            path.relative_to(dependency_dir)
            actual_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        except (ValueError, OSError):
            return None
        if actual_hash != expected_hash:
            return None
        resolved[name] = path

    prepared = RustDependencies(
        key=key,
        directory=dependency_dir,
        engine_rlib=resolved["engine_rlib"],
        pyo3_rlib=resolved["pyo3_rlib"],
    )
    _PREPARED_DEPENDENCIES[memory_key] = prepared
    return prepared


def _prepare_dependencies(
    engine_root: Path,
    root: Path,
    *,
    engine_hash: str,
    build_identity: Mapping[str, Any],
    cargo_path: str,
    rustc_path: str,
    target_dir: Path,
) -> RustDependencies:
    """Compile the stable Cargo dependency graph once for parallel rustc consumers."""

    dependency_identity = _dependency_identity(build_identity)
    key = _dependency_key(build_identity)
    directory = _dependency_cache_dir(root, key)
    prepared = _read_prepared_dependencies(
        directory,
        key,
        expected_identity=dependency_identity,
        target_dir=target_dir,
    )
    if prepared is not None:
        return prepared

    locks = root / "locks"
    locks.mkdir(parents=True, exist_ok=True)
    with (locks / f"dependencies-{key}.lock").open("a+") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        prepared = _read_prepared_dependencies(
            directory,
            key,
            expected_identity=dependency_identity,
            target_dir=target_dir,
        )
        if prepared is not None:
            return prepared

        staged_engine = _stage_engine(engine_root, root, engine_hash)
        staged_lock = _artifact_lock_path(staged_engine)
        temp_root = root / "tmp"
        temp_root.mkdir(parents=True, exist_ok=True)
        workspace = Path(tempfile.mkdtemp(prefix=f"dependencies-{key}-", dir=temp_root))
        try:
            (workspace / "src").mkdir()
            (workspace / "src" / "lib.rs").write_text(
                "// NumSim dependency preparation crate; generated artifacts use rustc directly.\n"
            )
            (workspace / "Cargo.toml").write_text(
                _cargo_toml(
                    staged_engine,
                    "numsim_dependency_preparation",
                    generated_profile=build_identity["generated_profile"],
                    analysis_capable=build_identity.get("artifact_kind")
                    in {"synccheck", "racecheck"},
                    analysis_checker=(
                        build_identity.get("artifact_kind")
                        if build_identity.get("artifact_kind") in {"synccheck", "racecheck"}
                        else None
                    ),
                )
            )
            shutil.copy2(staged_lock, workspace / "Cargo.lock")
            completed = subprocess.run(
                [
                    cargo_path,
                    "build",
                    "--release",
                    "--locked",
                    "--message-format=json-render-diagnostics",
                ],
                cwd=workspace,
                env=_build_environment(
                    target_dir,
                    cargo_path=cargo_path,
                    rustc_path=rustc_path,
                    engine_hash=engine_hash,
                    build_identity=build_identity,
                ),
                capture_output=True,
                text=True,
            )
            build_log = completed.stdout + completed.stderr
            if completed.returncode != 0:
                raise NumSimBuildError(
                    "NumSim Rust dependency build failed:\n" + build_log[-12000:]
                )

            dependency_dir = target_dir / "release" / "deps"
            engine_rlib = _dependency_rlib(dependency_dir, "numsim_engine")
            pyo3_rlib = _dependency_rlib(dependency_dir, "pyo3")
            manifest = {
                "cache_key": key,
                "dependency_identity": dependency_identity,
                "generated_backend": _GENERATED_BACKEND,
                "engine_rlib": str(engine_rlib.relative_to(target_dir)),
                "engine_rlib_sha256": hashlib.sha256(engine_rlib.read_bytes()).hexdigest(),
                "pyo3_rlib": str(pyo3_rlib.relative_to(target_dir)),
                "pyo3_rlib_sha256": hashlib.sha256(pyo3_rlib.read_bytes()).hexdigest(),
            }
            write_dir = Path(tempfile.mkdtemp(prefix=f"dependencies-state-{key}-", dir=temp_root))
            candidate = write_dir / "dependencies"
            try:
                candidate.mkdir()
                (candidate / "manifest.json").write_text(
                    json.dumps(manifest, indent=2, sort_keys=True)
                )
                _remove_cache_entry(directory)
                directory.parent.mkdir(parents=True, exist_ok=True)
                os.replace(candidate, directory)
            finally:
                shutil.rmtree(write_dir, ignore_errors=True)
        finally:
            shutil.rmtree(workspace, ignore_errors=True)

    prepared = _read_prepared_dependencies(
        directory,
        key,
        expected_identity=dependency_identity,
        target_dir=target_dir,
    )
    if prepared is None:
        raise NumSimBuildError(
            f"Rust dependency cache write did not produce a valid entry: {directory}"
        )
    return prepared


def _compile_generated_artifact(
    source_path: Path,
    library_path: Path,
    *,
    lib_target: str,
    dependencies: RustDependencies,
    generated_profile: Mapping[str, Any],
    rustc_path: str,
    env: Mapping[str, str],
) -> subprocess.CompletedProcess[str]:
    compile_env = dict(env)
    # Stable rustc still gates the parallel query scheduler behind ``-Z``.
    compile_env["RUSTC_BOOTSTRAP"] = "1"
    lto = generated_profile.get("lto")
    lto_enabled = lto not in {None, "off"}
    rustc_args = [
        rustc_path,
        "--crate-name",
        lib_target,
        "--edition=2024",
        str(source_path),
        "--crate-type=cdylib",
        "--emit=link",
        "-C",
        f"opt-level={generated_profile['opt_level']}",
        "-C",
        "debug-assertions=off",
        "-C",
        "overflow-checks=off",
        "-C",
        f"embed-bitcode={'yes' if lto_enabled else 'no'}",
        "-C",
        f"codegen-units={generated_profile['codegen_units']}",
        "-Z",
        f"threads={generated_profile['rustc_threads']}",
    ]
    if lto is not None:
        rustc_args.extend(("-C", f"lto={lto}"))
    target_cpu = generated_profile.get("target_cpu")
    if target_cpu is not None:
        rustc_args.extend(("-C", f"target-cpu={target_cpu}"))
    strip = generated_profile.get("strip", "debuginfo")
    if strip is not None:
        rustc_args.extend(("-C", f"strip={strip}"))
    rustc_args.extend(
        (
            "-L",
            f"dependency={dependencies.directory}",
            "--extern",
            f"numsim_engine={dependencies.engine_rlib}",
            "--extern",
            f"pyo3={dependencies.pyo3_rlib}",
            "-o",
            str(library_path),
        )
    )
    return subprocess.run(
        rustc_args,
        cwd=source_path.parent.parent,
        env=compile_env,
        capture_output=True,
        text=True,
    )


def _remove_cache_entry(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.exists():
        shutil.rmtree(path)


def _materialize_artifact(
    shared_directory: Path,
    artifact_dir: Path,
    *,
    root: Path,
    cache_rejections: list[str],
) -> None:
    if shared_directory == artifact_dir:
        return

    manifest_text = (shared_directory / "manifest.json").read_text()
    manifest = json.loads(manifest_text)
    original_reasons = manifest.get("cache_rebuild_reasons")
    manifest["cache_rebuild_reasons"] = list(
        dict.fromkeys((*manifest.get("cache_rebuild_reasons", ()), *cache_rejections))
    )
    temp_root = root / "tmp"
    temp_root.mkdir(parents=True, exist_ok=True)
    materialize_dir = Path(
        tempfile.mkdtemp(prefix=f"materialize-{artifact_dir.name}-", dir=temp_root)
    )
    candidate = materialize_dir / "artifact"
    try:
        candidate.mkdir()
        shutil.copy2(shared_directory / "module.so", candidate / "module.so")
        shared_source = shared_directory / "src" / "lib.rs"
        if shared_source.is_file():
            (candidate / "src").mkdir()
            shutil.copy2(shared_source, candidate / "src" / "lib.rs")
        if manifest["cache_rebuild_reasons"] != original_reasons:
            manifest_text = json.dumps(manifest, indent=2, sort_keys=True)
        (candidate / "manifest.json").write_text(manifest_text)
        _remove_cache_entry(artifact_dir)
        artifact_dir.parent.mkdir(parents=True, exist_ok=True)
        os.replace(candidate, artifact_dir)
    finally:
        shutil.rmtree(materialize_dir, ignore_errors=True)


def build_artifact(
    spec: ModuleSpec,
    source_template: str,
    *,
    cache_dir: str | Path | None = None,
    prepared: PreparedArtifact,
) -> Artifact:
    if prepared.cache_kind != _GENERATED_CACHE_KIND:
        raise NumSimBuildError("prepared artifact is not a generated-source request")
    if cache_dir is not None and Path(cache_dir).expanduser().resolve() != prepared.root:
        raise NumSimBuildError("prepared artifact does not match the supplied cache directory")
    engine_root = prepared.engine_root
    root = prepared.root
    build_root = prepared.build_root
    cargo_path = prepared.cargo_path
    rustc_path = prepared.rustc_path
    cargo_identity = prepared.cargo_identity
    rustc_identity = prepared.rustc_identity
    rust_target = prepared.rust_target
    engine_hash = prepared.engine_hash
    build_config = prepared.build_config
    generated_profile = prepared.generated_profile
    build_identity = prepared.build_identity
    target_dir = prepared.target_dir
    key = prepared.key
    module_name = prepared.module_name
    lib_target = prepared.lib_target
    artifact_dir = prepared.artifact_dir
    shared_artifact_dir = prepared.shared_artifact_dir
    cache_kind = prepared.cache_kind
    codegen_fingerprint = prepared.codegen_fingerprint
    source_identity = prepared.generated_source_identity

    rendered_module_name, source = _render_generated_source(source_template, key)
    if rendered_module_name != module_name:
        raise NumSimBuildError("artifact module name does not match its cache key")

    def read_cached(
        directory: Path,
        diagnostics: list[str] | None = None,
        *,
        reuse_from: Artifact | None = None,
    ) -> Artifact | None:
        return _read_cached(
            directory,
            key,
            source,
            expected_build_identity=build_identity,
            diagnostics=diagnostics,
            reuse_from=reuse_from,
            expected_cache_kind=cache_kind,
            expected_codegen_fingerprint=codegen_fingerprint,
            expected_generated_source_identity=source_identity,
        )

    def validate_cached(cached: Artifact) -> Artifact:
        if cached.spec is None or cached.spec.to_manifest(
            include_source_spans=False
        ) != spec.to_manifest(include_source_spans=False):
            raise NumSimBuildError(
                "generated module spec changed without changing its artifact cache identity"
            )
        return cached

    cache_rejections: list[str] = []
    cached = read_cached(artifact_dir, cache_rejections)
    if cached is not None:
        return validate_cached(cached)

    local_locks = root / "locks"
    local_locks.mkdir(parents=True, exist_ok=True)
    with (local_locks / f"{key}.lock").open("a+") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        cached = read_cached(artifact_dir, cache_rejections)
        if cached is not None:
            return validate_cached(cached)

        shared_rejections: list[str] = []
        shared = read_cached(shared_artifact_dir, shared_rejections)
        if shared is None:
            shared_locks = build_root / "locks"
            shared_locks.mkdir(parents=True, exist_ok=True)
            with (shared_locks / f"artifact-{key}.lock").open("a+") as shared_lock_file:
                fcntl.flock(shared_lock_file.fileno(), fcntl.LOCK_EX)
                shared = read_cached(shared_artifact_dir, shared_rejections)
                if shared is None:
                    _remove_cache_entry(shared_artifact_dir)
                    temp_root = build_root / "tmp"
                    temp_root.mkdir(parents=True, exist_ok=True)
                    build_dir = Path(tempfile.mkdtemp(prefix=f"{key}-", dir=temp_root))
                    try:
                        dependencies = _prepare_dependencies(
                            engine_root,
                            build_root,
                            engine_hash=engine_hash,
                            build_identity=build_identity,
                            cargo_path=cargo_path,
                            rustc_path=rustc_path,
                            target_dir=target_dir,
                        )
                        (build_dir / "src").mkdir()
                        (build_dir / "src" / "lib.rs").write_text(source)
                        with ThreadPoolExecutor(max_workers=1) as compiler:
                            compilation = compiler.submit(
                                _compile_generated_artifact,
                                build_dir / "src" / "lib.rs",
                                build_dir / "module.so",
                                lib_target=lib_target,
                                dependencies=dependencies,
                                generated_profile=generated_profile,
                                rustc_path=rustc_path,
                                env=_build_environment(
                                    target_dir,
                                    cargo_path=cargo_path,
                                    rustc_path=rustc_path,
                                    engine_hash=engine_hash,
                                    build_identity=build_identity,
                                ),
                            )
                            # This work is independent of rustc. Normalize exactly as
                            # the on-disk JSON does, then retain the parser's verified spec.
                            spec_manifest = json.loads(json.dumps(spec.to_manifest()))
                            try:
                                restored_spec = module_spec_from_manifest(spec_manifest)
                            except (TypeError, ValueError):
                                # Preserve the normal cache rejection path for invalid metadata.
                                restored_spec = None
                            completed = compilation.result()
                        build_log = completed.stdout + completed.stderr
                        (build_dir / "build.log").write_text(build_log)
                        if completed.returncode != 0:
                            raise NumSimBuildError(
                                f"Generated Rust build failed for {key}:\n{build_log[-12000:]}"
                            )
                        library_sha256 = hashlib.sha256(
                            (build_dir / "module.so").read_bytes()
                        ).hexdigest()
                        manifest = {
                            "cache_key": key,
                            "module_name": module_name,
                            **abi_metadata(),
                            "cache_rebuild_reasons": list(dict.fromkeys(shared_rejections)),
                            "build_identity": build_identity,
                            "engine_hash": engine_hash,
                            "pyo3": PYO3_VERSION,
                            "python_cache_tag": sys.implementation.cache_tag,
                            "python_executable": str(Path(sys.executable).resolve()),
                            "numpy_version": np.__version__,
                            "profile": build_config.profile_enabled,
                            "cargo": cargo_identity,
                            "cargo_path": cargo_path,
                            "generated_backend": _GENERATED_BACKEND,
                            "rustc": rustc_identity,
                            "rustc_path": rustc_path,
                            "rust_target": rust_target,
                            "cache_kind": cache_kind,
                            "codegen_fingerprint": codegen_fingerprint,
                            "generated_source_identity": source_identity,
                            "spec": spec_manifest,
                            "spec_sha256": _json_sha256(spec_manifest),
                            "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
                            "library_sha256": library_sha256,
                        }
                        (build_dir / "manifest.json").write_text(
                            json.dumps(manifest, indent=2, sort_keys=True)
                        )
                        shared_artifact_dir.parent.mkdir(parents=True, exist_ok=True)
                        os.replace(build_dir, shared_artifact_dir)
                    except Exception:
                        shutil.rmtree(build_dir, ignore_errors=True)
                        raise

                    shared = read_cached(
                        shared_artifact_dir,
                        reuse_from=Artifact(
                            key=key,
                            module_name=module_name,
                            directory=shared_artifact_dir,
                            library_path=shared_artifact_dir / "module.so",
                            source=source,
                            manifest=manifest,
                            spec=restored_spec,
                        ),
                    )
                    if shared is None:
                        raise NumSimBuildError(
                            "Shared artifact cache write did not produce a valid entry: "
                            f"{shared_artifact_dir}"
                        )

        _materialize_artifact(
            shared.directory,
            artifact_dir,
            root=root,
            cache_rejections=cache_rejections,
        )

    artifact = read_cached(artifact_dir, reuse_from=shared)
    if artifact is None:
        raise NumSimBuildError(
            f"Artifact cache materialization did not produce a valid entry: {artifact_dir}"
        )
    return validate_cached(artifact)
