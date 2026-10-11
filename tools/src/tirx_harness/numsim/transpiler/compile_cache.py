"""Content-addressed cache for one native NumSim compile.

The native frontend analyzes a kernel sequence and emits its generated Rust in
one call, so a module manifest and a module source are two halves of the same
compile.  This cache stores them together under the source IR, the identity of
the code that produced them, and the emission mode; a hit restores the spec
from the stored manifest and serves the stored source without compiling again.

`codegen_dependency_paths` and `split_configuration_identity` state what code
generation depends on, which the compiled-artifact cache (`build`) reuses for
its own key so the two cannot disagree about when generated Rust changed.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import shutil
import sys
import tempfile
from functools import cache
from pathlib import Path
from typing import Any, Literal

import tvm
import tvm_ffi

from . import suspend_scaffold
from ..abi import NUMSIM_ABI_VERSION
from ..errors import NumSimBuildError
from .artifact_template import compile_native_module, render_native_module
from .build import artifact_kind, generated_source_identity
from .cache import default_cache_root
from .frontend import (
    ModuleSpec,
    attach_source_nodes,
    module_spec_from_manifest,
    native_module_spec,
    source_kernels,
    verify,
)

# NumSim modules outside ``transpiler/`` that participate in generated Rust.
# The registry test pins this list to the transpiler's actual static import
# closure.  Both this cache and the compiled-artifact cache consume the paths
# below so they cannot disagree about when code generation changed.
_CODEGEN_DEPENDENCY_MODULES = (
    "abi.py",
    "dtype_abi.py",
    "errors.py",
)


def codegen_dependency_paths() -> tuple[Path, ...]:
    from .native_frontend import library_path

    numsim_root = Path(__file__).resolve().parent.parent
    paths = list((numsim_root / "transpiler").rglob("*.py"))
    paths.append(library_path())
    paths.append(numsim_root / "dtype_registry.json")
    paths.extend((numsim_root / name) for name in _CODEGEN_DEPENDENCY_MODULES)
    try:
        import tvm
    except ImportError:
        pass
    else:
        tvm_root = Path(tvm.__file__).resolve().parent
        paths.extend((tvm_root / name) for name in ("__init__.py", "base.py"))
        paths.extend((tvm_root / "tirx").rglob("*.py"))
        paths.extend((tvm_root / "ir").rglob("*.py"))
        paths.extend((tvm_root / "arith").rglob("*.py"))
    try:
        import tvm_ffi
    except ImportError:
        pass
    else:
        paths.extend(Path(tvm_ffi.__file__).resolve().parent.rglob("*.py"))
    return tuple(sorted(set(paths)))


@cache
def codegen_dependency_source_digest() -> str:
    digest = hashlib.sha256()
    paths = codegen_dependency_paths()
    if paths:
        common_root = Path(os.path.commonpath(paths))
        for path in paths:
            if not path.is_file():
                continue
            digest.update(str(path.relative_to(common_root)).encode())
            digest.update(b"\0")
            digest.update(path.read_bytes())
            digest.update(b"\0")
    return digest.hexdigest()


def split_configuration_identity() -> dict[str, int]:
    return suspend_scaffold.thresholds()


def _ensure_cache_root(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    metadata = root.stat()
    if metadata.st_uid != os.getuid():
        raise NumSimBuildError(f"NumSim compile cache is not owned by the current user: {root}")
    if metadata.st_mode & 0o022:
        raise NumSimBuildError(f"NumSim compile cache must not be group- or world-writable: {root}")


def _tvm_library_identity() -> list[dict[str, Any]]:
    from tvm.base import _LOADED_LIBS

    libraries = {Path(tvm_ffi.LIB._name).resolve()}
    libraries.update(Path(library._name).resolve() for library in _LOADED_LIBS.values())
    result = []
    for library in sorted(libraries):
        metadata = library.stat()
        result.append(
            {
                "path": str(library),
                "size": metadata.st_size,
                "mtime_ns": metadata.st_mtime_ns,
            }
        )
    return result


def _compile_identity(source_identity: str, *, kind: str) -> dict[str, Any]:
    """What one stored compile was produced from.

    ``transpiler_source`` digests the native frontend library together with
    every Python source that reaches it, including TVM's own TIRx package.
    """

    return {
        "source": source_identity,
        "artifact_kind": kind,
        "numsim_abi_version": NUMSIM_ABI_VERSION,
        "transpiler_source": codegen_dependency_source_digest(),
        "split_configuration": split_configuration_identity(),
        "python_cache_tag": sys.implementation.cache_tag,
        "tvm_version": tvm.__version__,
        "tvm_library": _tvm_library_identity(),
    }


def _cache_key(identity: dict[str, Any]) -> str:
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _read_cached(
    directory: Path,
    key: str,
    identity: dict[str, Any],
    funcs: tuple[Any, ...],
) -> tuple[ModuleSpec, str] | None:
    """One stored compile bound to ``funcs``, or ``None`` when it cannot serve them."""

    manifest_path = directory / "manifest.json"
    spec_path = directory / "spec.json"
    source_path = directory / "module.rs"
    if not (manifest_path.is_file() and spec_path.is_file() and source_path.is_file()):
        return None
    try:
        manifest = json.loads(manifest_path.read_text())
        payload = spec_path.read_text()
        source = source_path.read_text()
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return None
    if manifest.get("cache_key") != key or manifest.get("identity") != identity:
        return None
    if manifest.get("spec_sha256") != hashlib.sha256(payload.encode()).hexdigest():
        return None
    if manifest.get("source_sha256") != hashlib.sha256(source.encode()).hexdigest():
        return None
    try:
        spec = module_spec_from_manifest(json.loads(payload))
        # Binding the restored manifest to the kernels in hand also checks each
        # one's structural hash and operation sequence against it.
        return attach_source_nodes(spec, funcs, _render_script=False, _cache_key=key), source
    except (json.JSONDecodeError, ValueError):
        return None


def _remove_cache_entry(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.exists():
        shutil.rmtree(path)


def _write_cached(
    root: Path,
    directory: Path,
    key: str,
    identity: dict[str, Any],
    payload: str,
    source: str,
) -> None:
    temp_root = root / "tmp"
    temp_root.mkdir(parents=True, exist_ok=True)
    write_dir = Path(tempfile.mkdtemp(prefix=f"{key}-", dir=temp_root))
    candidate = write_dir / "entry"
    try:
        candidate.mkdir()
        (candidate / "spec.json").write_text(payload)
        (candidate / "module.rs").write_text(source)
        manifest = {
            "cache_key": key,
            "identity": identity,
            "spec_sha256": hashlib.sha256(payload.encode()).hexdigest(),
            "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
        }
        (candidate / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
        _remove_cache_entry(directory)
        directory.parent.mkdir(parents=True, exist_ok=True)
        os.replace(candidate, directory)
    finally:
        shutil.rmtree(write_dir, ignore_errors=True)


def compile_module_cached(
    source: Any,
    *,
    precision: str = "native",
    analysis_capable: bool,
    analysis_checker: Literal["synccheck", "racecheck"] | None,
) -> tuple[ModuleSpec, str]:
    """Return one compile's verified module spec and generated module source.

    A stored entry is written only after the spec verified and the module
    rendered, so a hit serves a spec that already passed both.  On a miss
    `verify` precedes emission, whose own host-ABI check precedes the native
    frontend's emission failure.
    """

    funcs = source_kernels(source)
    identity = _compile_identity(
        generated_source_identity(funcs, precision=precision),
        kind=artifact_kind(analysis_capable=analysis_capable, analysis_checker=analysis_checker),
    )
    key = _cache_key(identity)
    root = default_cache_root() / "compile"
    _ensure_cache_root(root)
    directory = root / "entries" / key
    cached = _read_cached(directory, key, identity, funcs)
    if cached is not None:
        return cached

    locks = root / "locks"
    locks.mkdir(parents=True, exist_ok=True)
    with (locks / f"{key}.lock").open("a+") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        cached = _read_cached(directory, key, identity, funcs)
        if cached is not None:
            return cached

        compiled = compile_native_module(
            funcs,
            precision=precision,
            analysis_capable=analysis_capable,
            analysis_checker=analysis_checker,
        )
        spec = native_module_spec(funcs, compiled[0], compiled[1], compiled[2], render_script=False)
        verify(spec)
        generated = render_native_module(compiled, spec)
        _write_cached(
            root,
            directory,
            key,
            identity,
            json.dumps(spec.to_manifest(), sort_keys=True, separators=(",", ":")),
            generated,
        )
    return spec, generated
