"""The artifact cache key must hash exactly the transpiler's own dependencies.

Hashing an unrelated module rebuilds every cached artifact whenever that module
is edited; omitting a real dependency serves a stale artifact after a codegen
change. Both failures are silent, so the dependency set is pinned here against
the transpiler package's actual static import closure.
"""

from __future__ import annotations

import ast
from pathlib import Path

from tirx_harness.numsim.transpiler import build as artifact_build
from tirx_harness.numsim.transpiler.compile_cache import codegen_dependency_paths

_NUMSIM_ROOT = Path(artifact_build.__file__).resolve().parent.parent
_TRANSPILER_ROOT = _NUMSIM_ROOT / "transpiler"


def _transpiler_sources() -> list[Path]:
    return sorted(_TRANSPILER_ROOT.rglob("*.py"))


def _package_parts(path: Path) -> list[str]:
    """Return the package path of ``path`` as parts relative to ``numsim``."""

    parts = list(path.resolve().relative_to(_NUMSIM_ROOT).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    return parts[:-1] if parts else []


def _relative_import_targets(path: Path) -> set[tuple[str, ...]]:
    """Resolve every ``from ..x import y`` in ``path`` to numsim-relative parts."""

    tree = ast.parse(path.read_text(), filename=str(path))
    package = _package_parts(path)
    targets: set[tuple[str, ...]] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom) or not node.level:
            continue
        # level 1 is the containing package; each extra level climbs one more.
        ascend = node.level - 1
        base = package[: len(package) - ascend] if ascend else list(package)
        if ascend > len(package):
            continue  # escapes numsim entirely; covered by the absolute-import guard
        targets.add(tuple(base + (node.module.split(".") if node.module else [])))
    return targets


def _numsim_root_dependency_closure() -> set[str]:
    """Return the numsim-root modules the transpiler package imports, transitively."""

    pending = list(_transpiler_sources())
    seen_files = {path.resolve() for path in pending}
    root_modules: set[str] = set()
    while pending:
        for target in _relative_import_targets(pending.pop()):
            if len(target) != 1 or target[0] == "transpiler":
                continue
            module_path = _NUMSIM_ROOT / f"{target[0]}.py"
            if not module_path.exists() or module_path in seen_files:
                continue
            seen_files.add(module_path)
            root_modules.add(module_path.name)
            pending.append(module_path)
    return root_modules


def test_transpiler_has_no_absolute_intra_package_imports() -> None:
    """The closure above only follows relative imports, so absolute ones would hide."""

    offenders = []
    for path in _transpiler_sources():
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and not node.level and node.module:
                names = [node.module]
            offenders.extend(
                (path.name, name) for name in names if name.split(".")[0] == "tirx_harness"
            )
    assert offenders == []


def test_codegen_key_hashes_exactly_the_transpiler_dependency_closure() -> None:
    hashed = {
        path.name
        for path in codegen_dependency_paths()
        if path.parent == _NUMSIM_ROOT and path.suffix == ".py"
    }

    assert hashed == _numsim_root_dependency_closure()


def test_codegen_key_drops_modules_the_transpiler_never_imports() -> None:
    paths = codegen_dependency_paths()
    names = {str(path) for path in paths}

    assert str(_NUMSIM_ROOT / "api.py") not in names
    assert str(_NUMSIM_ROOT / "bindings.py") not in names
    assert str(_NUMSIM_ROOT / "host_abi.py") not in names


def test_codegen_key_still_hashes_every_transpiler_source() -> None:
    hashed = set(codegen_dependency_paths())

    assert {path.resolve() for path in _transpiler_sources()} <= hashed


def test_native_frontend_binary_invalidates_all_compile_caches(tmp_path, monkeypatch):
    from tirx_harness.numsim.transpiler import compile_cache, native_frontend

    library = native_frontend.library_path()
    assert library in codegen_dependency_paths()
    replacement = tmp_path / library.name
    replacement.write_bytes(library.read_bytes())
    monkeypatch.setattr(native_frontend, "library_path", lambda: replacement)

    def clear_digests():
        compile_cache.codegen_dependency_source_digest.cache_clear()
        artifact_build._static_codegen_fingerprint.cache_clear()

    def identities():
        clear_digests()
        return (
            compile_cache._compile_identity("source", kind="numsim"),
            artifact_build._codegen_fingerprint(),
        )

    try:
        before = identities()
        with replacement.open("ab") as output:
            output.write(b"changed-frontend-build")
        after = identities()
        assert all(old != new for old, new in zip(before, after, strict=True))
    finally:
        clear_digests()
