from __future__ import annotations

import hashlib
import json
import re
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from tirx_harness.numsim import api as numsim_api
from tirx_harness.numsim.errors import NumSimBuildError
from tirx_harness.numsim.errors import NumSimExecutionError
from tirx_harness.numsim.abi import abi_metadata
from tirx_harness.numsim.transpiler import build, cache
from tirx_harness.numsim.transpiler.frontend import analyze

from tests.numsim.support.kernels import lane_add, no_op_kernel


def _generated_key(build_identity: dict[str, object]) -> bytes:
    """Key payload for a fixed generated source, varying only the build identity."""

    return build._generated_key_payload(
        build_identity,
        codegen_fingerprint="codegen-fingerprint",
        generated_source_identity="workspace-lock-source",
        artifact_kind="numsim",
    )


def _build_generated(source, template: str, cache_dir: Path) -> build.Artifact:
    """Build a generated artifact through the sole production path.

    `prepare_generated_artifact` resolves the cache roots and the content
    addressed key; `build_artifact` then only consumes that preparation.
    """

    prepared = build.prepare_generated_artifact(source, cache_dir=cache_dir)
    return build.build_artifact(
        analyze(source),
        template,
        cache_dir=cache_dir,
        prepared=prepared,
    )


def _test_build_identity(engine_hash: str = "engine-hash") -> dict[str, object]:
    return {
        "engine_hash": engine_hash,
        **abi_metadata(),
        "generated_profile": build._generated_profile(""),
        "toolchain": {"rust_target": "test-target", "features": ["python"]},
    }


def _test_manifest(source: str, library: Path) -> dict[str, object]:
    identity = _test_build_identity()
    return {
        "cache_key": "key",
        "module_name": "module",
        **abi_metadata(),
        "cache_rebuild_reasons": [],
        "build_identity": identity,
        "engine_hash": identity["engine_hash"],
        "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "library_sha256": hashlib.sha256(library.read_bytes()).hexdigest(),
    }


def _test_native_metadata() -> dict[str, object]:
    identity = _test_build_identity()
    return {
        "cache_key": "key",
        "build_identity": identity,
        "engine_hash": identity["engine_hash"],
        **abi_metadata(),
    }


def _mock_build_environment(monkeypatch, tmp_path):
    engine_root = tmp_path / "engine"
    shared_cache_root = tmp_path / "shared-cache"
    lock = engine_root / "artifact-template" / "Cargo.lock"
    lock.parent.mkdir(parents=True)
    (engine_root / "Cargo.toml").write_text('[package]\nname = "engine"\n')
    lock.write_text("locked-dependencies")
    monkeypatch.setattr(build, "_engine_root", lambda: engine_root)
    monkeypatch.setattr(build, "hash_engine_build_inputs", lambda _root: "engine-hash")
    monkeypatch.setattr(build, "cargo_version", lambda *_args: "cargo-test")
    monkeypatch.setattr(build, "rustc_version", lambda *_args: "rustc-test")
    monkeypatch.setattr(build, "rustc_host_target", lambda *_args: "test-target")
    monkeypatch.setattr(build, "rust_tool", lambda name: f"/tool/{name}")
    monkeypatch.setattr(build, "default_cache_root", lambda: shared_cache_root)
    monkeypatch.setattr(build, "default_build_cache_root", lambda: shared_cache_root / "build")
    build._PREPARED_DEPENDENCIES.clear()
    return lock


def _fake_dependency_build(args, *, env):
    assert args[0:2] == ["/tool/cargo", "build"]
    assert "--locked" in args
    target_dir = Path(env["CARGO_TARGET_DIR"])
    dependency_dir = target_dir / "release" / "deps"
    dependency_dir.mkdir(parents=True, exist_ok=True)
    (dependency_dir / "libnumsim_engine-test.rlib").write_bytes(b"engine")
    (dependency_dir / "libpyo3-test.rlib").write_bytes(b"pyo3")
    return subprocess.CompletedProcess(args, 0, stdout="", stderr="")


def _direct_source(args) -> Path:
    return Path(args[args.index("--edition=2024") + 1])


def _direct_output(args) -> Path:
    return Path(args[args.index("-o") + 1])


def test_build_identity_records_the_numsim_abi_version(tmp_path):
    identity = build._build_identity(
        "engine-hash",
        cache_root=tmp_path,
        generated_profile=build._generated_profile(""),
        cargo_path="/tool/cargo",
        rustc_path="/tool/rustc",
        cargo_identity="cargo-test",
        rustc_identity="rustc-test",
        rust_target="test-target",
    )

    for name, expected in abi_metadata().items():
        assert identity[name] == expected
    assert identity["generated_profile"] == {
        "opt_level": build._GENERATED_OPT_LEVEL,
        "codegen_units": build._GENERATED_CODEGEN_UNITS,
        "rustc_threads": build._GENERATED_RUSTC_THREADS,
        "lto": build._GENERATED_ARTIFACT_RUSTC_LTO,
        "target_cpu": build._GENERATED_ARTIFACT_TARGET_CPU,
    }
    assert identity["artifact_kind"] == "numsim"

    analysis_identity = build._build_identity(
        "engine-hash",
        cache_root=tmp_path,
        generated_profile=build._generated_profile(""),
        analysis_capable=True,
        cargo_path="/tool/cargo",
        rustc_path="/tool/rustc",
        cargo_identity="cargo-test",
        rustc_identity="rustc-test",
        rust_target="test-target",
    )
    assert analysis_identity["artifact_kind"] == "synccheck"


def test_generated_profile_keeps_all_source_sizes_at_o3():
    assert build._generated_profile("") == build._generated_profile("x" * 2_000_000)
    assert build._generated_profile("")["opt_level"] == 3


def test_prepare_analysis_artifact_keeps_o3_default(monkeypatch, tmp_path):
    _mock_build_environment(monkeypatch, tmp_path)
    monkeypatch.delenv(build._GENERATED_OPT_LEVEL_ENV, raising=False)
    monkeypatch.setattr(build, "generated_source_identity", lambda _source: "source-identity")
    monkeypatch.setattr(build, "_codegen_fingerprint", lambda: "codegen-fingerprint")

    prepared = build.prepare_generated_artifact(
        no_op_kernel,
        cache_dir=tmp_path / "artifacts",
        analysis_capable=True,
        analysis_checker="synccheck",
    )

    assert prepared.build_config.release_opt_level == 3
    assert prepared.generated_profile["opt_level"] == 3


def test_analysis_generated_profile_uses_more_codegen_units_and_frontend_lto_policy():
    profile = build._generated_profile(
        "",
        build_config=build._generated_build_config(analysis_capable=True),
    )

    assert profile == {
        "opt_level": 3,
        "codegen_units": 32,
        "rustc_threads": 8,
        "lto": "off",
        "target_cpu": "native",
    }


def test_racecheck_generated_profile_uses_runtime_optimized_codegen_units():
    profile = build._generated_profile(
        "",
        build_config=build._generated_build_config(
            analysis_capable=True,
            analysis_checker="racecheck",
        ),
    )

    assert profile == {
        "opt_level": 3,
        "codegen_units": 6,
        "rustc_threads": 8,
        "lto": "off",
        "target_cpu": "native",
    }


@pytest.mark.parametrize("checker", ("synccheck", "racecheck"))
def test_analysis_o0_profile_uses_the_shared_debug_partition(checker, monkeypatch):
    monkeypatch.setenv(build._GENERATED_OPT_LEVEL_ENV, "0")

    profile = build._generated_profile(
        "",
        build_config=build._generated_build_config(
            analysis_capable=True,
            analysis_checker=checker,
        ),
    )

    assert profile == {
        "opt_level": 0,
        "codegen_units": 32,
        "rustc_threads": 8,
        "lto": "off",
    }


def test_analysis_checker_requires_analysis_capable_codegen():
    with pytest.raises(ValueError, match="requires analysis-capable"):
        build._generated_build_config(analysis_checker="racecheck")


def test_generated_profile_changes_artifacts_without_splitting_dependency_target(tmp_path):
    default_identity = _test_build_identity()
    alternate_identity = deepcopy(default_identity)
    alternate_identity["generated_profile"] = {
        "opt_level": 1,
        "codegen_units": build._GENERATED_CODEGEN_UNITS * 2,
        "rustc_threads": build._GENERATED_RUSTC_THREADS,
        "lto": build._GENERATED_ARTIFACT_RUSTC_LTO,
        "target_cpu": build._GENERATED_ARTIFACT_TARGET_CPU,
    }

    assert _generated_key(default_identity) != _generated_key(alternate_identity)
    assert build._cargo_target_dir(tmp_path, default_identity) == build._cargo_target_dir(
        tmp_path, alternate_identity
    )


def test_link_profile_changes_split_dependency_target(tmp_path):
    default_identity = _test_build_identity()
    alternate_identity = deepcopy(default_identity)
    alternate_identity["generated_profile"] = {
        **alternate_identity["generated_profile"],
        "lto": "thin",
        "target_cpu": "native",
    }

    assert build._cargo_target_dir(tmp_path, default_identity) != build._cargo_target_dir(
        tmp_path, alternate_identity
    )


def test_racecheck_profile_precompiles_engine_and_targets_host(tmp_path):
    generated_profile = build._generated_profile(
        "",
        build_config=build._generated_build_config(
            analysis_capable=True,
            analysis_checker="racecheck",
        ),
    )
    manifest = build._cargo_toml(
        tmp_path,
        "numsim_dependency_preparation",
        generated_profile=generated_profile,
        analysis_capable=True,
    )
    identity = _test_build_identity()
    identity["generated_profile"] = generated_profile
    env = build._build_environment(
        tmp_path / "target",
        cargo_path="/tool/cargo",
        rustc_path="/tool/rustc",
        engine_hash="engine-hash",
        build_identity=identity,
    )

    assert generated_profile["lto"] == "off"
    assert '[profile.release]\nlto = "off"' in manifest
    assert env["CARGO_ENCODED_RUSTFLAGS"] == "-C\x1ftarget-cpu=native"


@pytest.mark.parametrize("field", sorted(abi_metadata()))
def test_cached_manifest_rejects_a_numsim_abi_mismatch(tmp_path, field):
    directory = tmp_path / "artifact"
    directory.mkdir()
    source = "generated source"
    library = directory / "module.so"
    library.write_bytes(b"native artifact")
    manifest = _test_manifest(source, library)
    manifest[field] = f"wrong-{manifest[field]}"
    (directory / "manifest.json").write_text(json.dumps(manifest))

    diagnostics: list[str] = []
    assert (
        build._read_cached(
            directory,
            "key",
            source,
            expected_build_identity=_test_build_identity(),
            diagnostics=diagnostics,
        )
        is None
    )
    assert len(diagnostics) == 1
    assert field in diagnostics[0]



def test_reused_artifact_metadata_observes_disk_changes_and_library_corruption(tmp_path):
    source = "generated source"
    library = tmp_path / "module.so"
    library.write_bytes(b"native artifact")
    spec = analyze(no_op_kernel)
    manifest = _test_manifest(source, library)
    manifest.update(
        cache_kind=build._GENERATED_CACHE_KIND,
        spec=spec.to_manifest(),
        spec_sha256=build._json_sha256(spec.to_manifest()),
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))

    def read(reuse_from=None):
        return build._read_cached(
            tmp_path,
            "key",
            source,
            expected_build_identity=_test_build_identity(),
            expected_cache_kind=build._GENERATED_CACHE_KIND,
            reuse_from=reuse_from,
        )

    first = read()
    assert first is not None
    repeated = read(first)
    assert repeated is not None and repeated.spec is first.spec
    manifest["spec"]["kernels"][0]["name"] = "changed_on_disk"
    manifest_path.write_text(json.dumps(manifest))
    assert read(first) is None
    manifest["spec"] = first.manifest["spec"]
    manifest_path.write_text(json.dumps(manifest))
    library.write_bytes(b"corrupt")
    assert read(first) is None


def test_cached_manifest_diagnoses_nested_build_identity_mismatch(tmp_path):
    directory = tmp_path / "artifact"
    directory.mkdir()
    source = "generated source"
    library = directory / "module.so"
    library.write_bytes(b"native artifact")
    manifest = _test_manifest(source, library)
    manifest["build_identity"]["toolchain"]["rust_target"] = "tampered-target"
    (directory / "manifest.json").write_text(json.dumps(manifest))

    diagnostics: list[str] = []
    assert (
        build._read_cached(
            directory,
            "key",
            source,
            expected_build_identity=_test_build_identity(),
            diagnostics=diagnostics,
        )
        is None
    )
    assert diagnostics == [
        "build_identity.toolchain.rust_target mismatch: "
        "cached='tampered-target', current='test-target'"
    ]


def test_cached_manifest_rejects_top_level_engine_hash_disagreement(tmp_path):
    directory = tmp_path / "artifact"
    directory.mkdir()
    source = "generated source"
    library = directory / "module.so"
    library.write_bytes(b"native artifact")
    manifest = _test_manifest(source, library)
    manifest["engine_hash"] = "tampered-engine"
    (directory / "manifest.json").write_text(json.dumps(manifest))

    diagnostics: list[str] = []
    assert (
        build._read_cached(
            directory,
            "key",
            source,
            expected_build_identity=_test_build_identity(),
            diagnostics=diagnostics,
        )
        is None
    )
    assert diagnostics == [
        "engine_hash disagrees with build_identity.engine_hash: "
        "engine_hash='tampered-engine', build_identity='engine-hash'"
    ]


def test_private_engine_hash_refactor_changes_only_the_content_addressed_key():
    before = _test_build_identity("engine-before")
    after = _test_build_identity("engine-after")

    before_key = hashlib.sha256(_generated_key(before)).hexdigest()
    after_key = hashlib.sha256(_generated_key(after)).hexdigest()

    assert before_key != after_key
    for name, expected in abi_metadata().items():
        assert before[name] == after[name] == expected


@pytest.fixture
def _loaded_artifact(monkeypatch, tmp_path):
    metadata = _test_native_metadata()
    native = SimpleNamespace(metadata=lambda: metadata)
    loader = SimpleNamespace(exec_module=lambda _module: None)
    spec = SimpleNamespace(loader=loader)
    monkeypatch.setattr(build.importlib.util, "spec_from_file_location", lambda *_args: spec)
    monkeypatch.setattr(build.importlib.util, "module_from_spec", lambda _spec: native)
    build._LOADED.clear()
    artifact = build.Artifact(
        key="key",
        module_name="module",
        directory=tmp_path,
        library_path=tmp_path / "module.so",
        source="source",
        manifest={
            "build_identity": _test_build_identity(),
            "engine_hash": "engine-hash",
        },
    )

    return artifact, metadata


@pytest.mark.parametrize("field", sorted(abi_metadata()))
@pytest.mark.parametrize("missing", [False, True])
def test_loaded_native_metadata_rejects_missing_or_mismatched_identity(
    _loaded_artifact, field, missing
):
    artifact, metadata = _loaded_artifact
    if missing:
        metadata.pop(field)
    else:
        metadata[field] = f"wrong-{metadata[field]}"

    with pytest.raises(NumSimBuildError, match=field):
        artifact.load()


def test_loaded_native_metadata_rejects_nested_build_identity_tampering(_loaded_artifact):
    artifact, metadata = _loaded_artifact
    metadata["build_identity"] = deepcopy(metadata["build_identity"])
    metadata["build_identity"]["toolchain"]["rust_target"] = "tampered-target"

    with pytest.raises(NumSimBuildError, match="build_identity.toolchain.rust_target"):
        artifact.load()


def test_loaded_native_metadata_rejects_top_level_engine_hash_disagreement(_loaded_artifact):
    artifact, metadata = _loaded_artifact
    metadata["engine_hash"] = "tampered-engine"

    with pytest.raises(
        NumSimBuildError,
        match="engine_hash disagrees with build_identity.engine_hash",
    ):
        artifact.load()


def test_execution_revalidates_native_metadata_after_module_load(tmp_path):
    manifest_identity = _test_build_identity()
    artifact = build.Artifact(
        key="key",
        module_name="module",
        directory=tmp_path,
        library_path=tmp_path / "module.so",
        source="source",
        manifest={
            "build_identity": manifest_identity,
            "engine_hash": "engine-hash",
        },
    )
    metadata = _test_native_metadata()
    metadata["build_identity"] = deepcopy(metadata["build_identity"])
    metadata["build_identity"]["toolchain"]["rust_target"] = "changed-after-load"
    native = SimpleNamespace(metadata=lambda: metadata)
    module = SimpleNamespace(artifact=artifact)

    with pytest.raises(
        NumSimExecutionError,
        match="metadata changed before execution.*build_identity.toolchain.rust_target",
    ):
        numsim_api._native_metadata_for_execution(module, native)


def test_parallel_artifacts_share_dependencies_without_sharing_generated_source(
    monkeypatch, tmp_path
):
    lock = _mock_build_environment(monkeypatch, tmp_path)
    state_lock = threading.Lock()
    active_builds = 0
    max_active_builds = 0
    dependency_builds = 0
    build_directories: set[Path] = set()
    build_targets: set[Path] = set()
    engine_snapshots: set[Path] = set()
    lib_targets: set[str] = set()

    def fake_run(args, *, cwd, env, capture_output, text):
        nonlocal active_builds, dependency_builds, max_active_builds
        del capture_output, text
        if args[0] == "/tool/cargo":
            dependency_builds += 1
            directory = Path(cwd)
            build_targets.add(Path(env["CARGO_TARGET_DIR"]))
            manifest = (directory / "Cargo.toml").read_text()
            engine_match = re.search(r'numsim-engine = \{ path = ("(?:[^"\\]|\\.)*")', manifest)
            assert engine_match is not None
            engine_snapshots.add(Path(json.loads(engine_match.group(1))))
            assert env["CARGO_INCREMENTAL"] == "0"
            assert env["RUSTC"] == "/tool/rustc"
            assert env["PATH"].split(":", 1)[0] == "/tool"
            assert env[build._ENGINE_HASH_ENV] == "engine-hash"
            embedded_identity = json.loads(env[build._BUILD_IDENTITY_ENV])
            assert embedded_identity["engine_hash"] == "engine-hash"
            assert embedded_identity["generated_backend"] == build._GENERATED_BACKEND
            assert (directory / "Cargo.lock").read_bytes() == lock.read_bytes()
            return _fake_dependency_build(args, env=env)

        assert args[0] == "/tool/rustc"
        with state_lock:
            active_builds += 1
            max_active_builds = max(max_active_builds, active_builds)
        try:
            directory = Path(cwd)
            lib_target = args[args.index("--crate-name") + 1]
            build_directories.add(directory)
            lib_targets.add(lib_target)
            assert env[build._ENGINE_HASH_ENV] == "engine-hash"
            embedded_identity = json.loads(env[build._BUILD_IDENTITY_ENV])
            assert embedded_identity["engine_hash"] == "engine-hash"
            assert embedded_identity["generated_backend"] == build._GENERATED_BACKEND
            source = _direct_source(args).read_bytes()
            time.sleep(0.05)
            assert _direct_source(args).read_bytes() == source
            library = _direct_output(args)
            library.parent.mkdir(parents=True, exist_ok=True)
            library.write_bytes(source)
            return subprocess.CompletedProcess(args, 0, stdout="", stderr="")
        finally:
            with state_lock:
                active_builds -= 1

    monkeypatch.setattr(build.subprocess, "run", fake_run)
    # Distinct kernels give distinct generated-source identities, so the two
    # artifacts get distinct cache keys and distinct cargo lib targets.
    requests = [
        (no_op_kernel, "module=__NUMSIM_MODULE__; key=__NUMSIM_CACHE_KEY__; variant=first"),
        (lane_add, "module=__NUMSIM_MODULE__; key=__NUMSIM_CACHE_KEY__; variant=second"),
    ]
    with ThreadPoolExecutor(max_workers=2) as executor:
        artifacts = list(
            executor.map(
                lambda item: _build_generated(
                    item[1][0], item[1][1], tmp_path / f"cache-{item[0]}"
                ),
                enumerate(requests),
            )
        )

    assert max_active_builds == 2
    assert dependency_builds == 1
    assert len(build_directories) == 2
    assert len(build_targets) == 1
    assert next(iter(build_targets)).parent == build._build_cache_root() / "cargo-target"
    assert engine_snapshots == {
        build._engine_snapshot_dir(build._build_cache_root(), "engine-hash")
    }
    assert len(lib_targets) == 2
    for artifact in artifacts:
        assert artifact.module_name in artifact.source
        assert artifact.library_path.read_bytes() == artifact.source.encode()
        assert artifact.manifest["cache_rebuild_reasons"] == [
            "cache entry is missing manifest.json or module.so"
        ]


def test_corrupt_local_library_is_restored_from_shared_artifact(monkeypatch, tmp_path):
    _mock_build_environment(monkeypatch, tmp_path)
    builds = 0

    def fake_run(args, *, cwd, env, capture_output, text):
        nonlocal builds
        del cwd, capture_output, text
        if args[0] == "/tool/cargo":
            return _fake_dependency_build(args, env=env)
        builds += 1
        library = _direct_output(args)
        library.parent.mkdir(parents=True, exist_ok=True)
        library.write_bytes(f"build-{builds}".encode())
        return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

    monkeypatch.setattr(build.subprocess, "run", fake_run)
    first = _build_generated(no_op_kernel, "__NUMSIM_MODULE__", tmp_path / "cache")
    first.library_path.write_bytes(b"corrupt")
    second = _build_generated(no_op_kernel, "__NUMSIM_MODULE__", tmp_path / "cache")

    assert builds == 1
    assert second.library_path.read_bytes() == b"build-1"



def test_cached_artifact_rejects_a_replaced_spec_with_stale_content_identity(
    monkeypatch, tmp_path
):
    _mock_build_environment(monkeypatch, tmp_path)
    spec = analyze(no_op_kernel)
    cached = SimpleNamespace(spec=spec)
    monkeypatch.setattr(build, "_read_cached", lambda *args, **kwargs: cached)
    prepared = build.prepare_generated_artifact(no_op_kernel, cache_dir=tmp_path / "cache")
    altered = replace(spec, kernels=(replace(spec.kernels[0], name="changed"),))
    with pytest.raises(NumSimBuildError, match="spec changed without changing"):
        build.build_artifact(altered, "__NUMSIM_MODULE__", prepared=prepared)


def test_identical_artifacts_in_distinct_cache_roots_compile_once(monkeypatch, tmp_path):
    _mock_build_environment(monkeypatch, tmp_path)
    builds = 0

    def fake_run(args, *, cwd, env, capture_output, text):
        nonlocal builds
        del cwd, capture_output, text
        if args[0] == "/tool/cargo":
            return _fake_dependency_build(args, env=env)
        builds += 1
        library = _direct_output(args)
        library.parent.mkdir(parents=True, exist_ok=True)
        library.write_bytes(b"shared-artifact")
        return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

    monkeypatch.setattr(build.subprocess, "run", fake_run)
    first = _build_generated(no_op_kernel, "__NUMSIM_MODULE__", tmp_path / "first")
    second = _build_generated(no_op_kernel, "__NUMSIM_MODULE__", tmp_path / "second")

    assert builds == 1
    assert first.key == second.key
    assert first.library_path != second.library_path
    assert first.library_path.read_bytes() == b"shared-artifact"
    assert second.library_path.read_bytes() == b"shared-artifact"


def test_corrupt_shared_artifact_is_recompiled_for_a_new_cache_root(monkeypatch, tmp_path):
    _mock_build_environment(monkeypatch, tmp_path)
    builds = 0

    def fake_run(args, *, cwd, env, capture_output, text):
        nonlocal builds
        del cwd, capture_output, text
        if args[0] == "/tool/cargo":
            return _fake_dependency_build(args, env=env)
        builds += 1
        library = _direct_output(args)
        library.parent.mkdir(parents=True, exist_ok=True)
        library.write_bytes(f"build-{builds}".encode())
        return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

    monkeypatch.setattr(build.subprocess, "run", fake_run)
    first = _build_generated(no_op_kernel, "__NUMSIM_MODULE__", tmp_path / "first")
    shared_library = build._build_cache_root() / "artifacts" / first.key / "module.so"
    shared_library.write_bytes(b"corrupt")

    second = _build_generated(no_op_kernel, "__NUMSIM_MODULE__", tmp_path / "second")

    assert builds == 2
    assert second.library_path.read_bytes() == b"build-2"


def test_build_rejects_group_or_world_writable_cache_root(monkeypatch, tmp_path):
    _mock_build_environment(monkeypatch, tmp_path)
    cache_root = tmp_path / "shared-cache"
    cache_root.mkdir()
    cache_root.chmod(0o777)

    with pytest.raises(NumSimBuildError, match="must not be group- or world-writable"):
        _build_generated(no_op_kernel, "__NUMSIM_MODULE__", cache_root)


def test_artifact_cache_override_does_not_split_the_shared_build_cache(monkeypatch, tmp_path):
    artifact_root = tmp_path / "artifact-cache"
    xdg_root = tmp_path / "xdg-cache"
    monkeypatch.setenv("NUMSIM_CACHE_DIR", str(artifact_root))
    monkeypatch.setenv("XDG_CACHE_HOME", str(xdg_root))
    monkeypatch.delenv("NUMSIM_BUILD_CACHE_DIR", raising=False)

    assert cache.default_cache_root() == artifact_root
    assert cache.default_build_cache_root() == xdg_root / "tirx-numsim" / "build"

    build_root = tmp_path / "explicit-build-cache"
    monkeypatch.setenv("NUMSIM_BUILD_CACHE_DIR", str(build_root))
    assert cache.default_build_cache_root() == build_root


def test_build_rejects_engine_changes_during_snapshot(monkeypatch, tmp_path):
    _mock_build_environment(monkeypatch, tmp_path)
    hashes = iter(("before", "after"))
    monkeypatch.setattr(build, "hash_engine_build_inputs", lambda _root: next(hashes))

    with pytest.raises(NumSimBuildError, match="changed while its build snapshot"):
        _build_generated(no_op_kernel, "__NUMSIM_MODULE__", tmp_path / "cache")


def _write_engine_hash_fixture(root: Path) -> None:
    (root / "src").mkdir(parents=True)
    (root / "fp-env" / "src").mkdir(parents=True)
    (root / "artifact-template").mkdir()
    (root / "Cargo.toml").write_text(
        """[package]
name = "engine"
version = "0.1.0"
edition = "2021"

[dependencies]
fp-env = { path = "fp-env" }
"""
    )
    (root / "src" / "lib.rs").write_text("pub fn engine() {}\n")
    (root / "fp-env" / "Cargo.toml").write_text(
        """[package]
name = "fp-env"
version = "0.1.0"
edition = "2021"
"""
    )
    (root / "fp-env" / "src" / "lib.rs").write_text("pub fn fp_env() {}\n")
    (root / "artifact-template" / "Cargo.lock").write_text("lock-v1\n")
    (root / "SUPPORTED_OPS.md").write_text("catalog-v1\n")


def test_cached_engine_snapshot_corruption_fails_closed(tmp_path):
    engine_root = tmp_path / "engine"
    build_root = tmp_path / "build-cache"
    _write_engine_hash_fixture(engine_root)
    build._ensure_cache_root(build_root)
    engine_hash = cache.hash_engine_build_inputs(engine_root)

    staged = build._stage_engine(engine_root, build_root, engine_hash)
    assert build._stage_engine(engine_root, build_root, engine_hash) == staged

    (staged / "src" / "lib.rs").write_text("pub fn corrupted_engine() {}\n")
    with pytest.raises(NumSimBuildError, match="does not match its content hash"):
        build._stage_engine(engine_root, build_root, engine_hash)


def test_engine_hash_covers_only_generated_artifact_build_inputs(tmp_path):
    engine_root = tmp_path / "engine"
    _write_engine_hash_fixture(engine_root)

    inputs = {
        path.relative_to(engine_root).as_posix()
        for path in cache.engine_build_input_paths(engine_root)
    }

    assert inputs == {
        "Cargo.toml",
        "artifact-template/Cargo.lock",
        "fp-env/Cargo.toml",
        "fp-env/src/lib.rs",
        "src/lib.rs",
    }


def test_repository_engine_hash_covers_runtime_rust_sources_but_not_tests_or_catalog():
    engine_root = build._engine_root().resolve()
    inputs = set(cache.engine_build_input_paths(engine_root))
    rust_sources = {
        path.resolve()
        for path in engine_root.rglob("*.rs")
        if not {"artifact-template", "target", "tests"} & set(path.relative_to(engine_root).parts)
    }

    assert rust_sources <= inputs
    assert not any("tests" in path.relative_to(engine_root).parts for path in inputs)
    assert (engine_root / "Cargo.toml").resolve() in inputs
    assert (engine_root / "artifact-template" / "Cargo.lock").resolve() in inputs
    assert (engine_root / "SUPPORTED_OPS.md").resolve() not in inputs


def test_engine_hash_ignores_catalog_docs_but_tracks_build_sources(tmp_path):
    engine_root = tmp_path / "engine"
    _write_engine_hash_fixture(engine_root)
    baseline = cache.hash_engine_build_inputs(engine_root)

    (engine_root / "SUPPORTED_OPS.md").write_text("catalog-v2\n")
    assert cache.hash_engine_build_inputs(engine_root) == baseline

    (engine_root / "src" / "lib.rs").write_text("pub fn changed_engine() {}\n")
    assert cache.hash_engine_build_inputs(engine_root) != baseline


@pytest.mark.parametrize(
    "relative_path",
    ["Cargo.toml", "fp-env/Cargo.toml", "fp-env/src/lib.rs", "artifact-template/Cargo.lock"],
)
def test_engine_hash_tracks_each_build_input_class(tmp_path, relative_path):
    engine_root = tmp_path / "engine"
    _write_engine_hash_fixture(engine_root)
    baseline = cache.hash_engine_build_inputs(engine_root)

    path = engine_root / relative_path
    path.write_text(path.read_text() + "# changed\n")

    assert cache.hash_engine_build_inputs(engine_root) != baseline


def test_rust_tool_resolves_a_relative_override_to_an_absolute_executable(monkeypatch, tmp_path):
    tool = tmp_path / "bin" / "cargo"
    tool.parent.mkdir()
    tool.write_text("#!/bin/sh\nexit 0\n")
    tool.chmod(0o755)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("NUMSIM_CARGO", "bin/cargo")

    resolved = cache.rust_tool("cargo")

    assert resolved == str(tool.resolve())
    assert Path(resolved).is_absolute()


def test_rust_tool_rejects_an_unresolvable_override(monkeypatch):
    monkeypatch.setenv("NUMSIM_CARGO", "missing-numsim-cargo")

    with pytest.raises(FileNotFoundError, match="cannot resolve"):
        cache.rust_tool("cargo")


def test_generated_artifact_rustc_flags_keep_o3_and_parallelize_queries(monkeypatch, tmp_path):
    captured: list[tuple[list[str], dict[str, str]]] = []

    def fake_run(args, *, cwd, env, capture_output, text):
        del cwd, capture_output, text
        captured.append((list(args), dict(env)))
        return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

    monkeypatch.setattr(build.subprocess, "run", fake_run)
    source_path = tmp_path / "generated" / "src" / "lib.rs"
    source_path.parent.mkdir(parents=True)
    source_path.write_text("// generated")
    dependencies = build.RustDependencies(
        key="deps-key",
        directory=tmp_path / "deps",
        engine_rlib=tmp_path / "deps" / "libnumsim_engine.rlib",
        pyo3_rlib=tmp_path / "deps" / "libpyo3.rlib",
    )
    build._compile_generated_artifact(
        source_path,
        tmp_path / "module.so",
        lib_target="numsim_artifact_test",
        dependencies=dependencies,
        generated_profile=build._generated_profile("x" * 2_000_000),
        rustc_path="/tool/rustc",
        env={},
    )

    ((argv, env),) = captured
    assert f"opt-level={build._GENERATED_OPT_LEVEL}" in argv
    assert argv[argv.index("-Z") + 1] == f"threads={build._GENERATED_RUSTC_THREADS}"
    assert "debug-assertions=off" in argv
    assert "overflow-checks=off" in argv
    assert f"lto={build._GENERATED_ARTIFACT_RUSTC_LTO}" in argv
    assert f"target-cpu={build._GENERATED_ARTIFACT_TARGET_CPU}" in argv
    assert "embed-bitcode=no" in argv
    assert env["RUSTC_BOOTSTRAP"] == "1"


def test_metadata_failure_waits_for_compiler_before_removing_workdir(monkeypatch, tmp_path):
    _mock_build_environment(monkeypatch, tmp_path)
    spec = analyze(no_op_kernel)
    prepared = build.prepare_generated_artifact(no_op_kernel, cache_dir=tmp_path / "local")
    compiler_started = threading.Event()
    metadata_failed = threading.Event()
    finish_compiler = threading.Event()
    directories = []

    def fake_run(args, *, cwd, env, capture_output, text):
        if args[0] == "/tool/cargo":
            return _fake_dependency_build(args, env=env)
        directory = Path(cwd)
        directories.append(directory)
        compiler_started.set()
        assert finish_compiler.wait(5)
        # Inputs must remain available until the compiler has finished.
        assert _direct_source(args).is_file()
        _direct_output(args).write_bytes(b"compiled")
        return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

    original_manifest = type(spec).to_manifest

    def fail_manifest(self, **kwargs):
        if self is spec:
            assert compiler_started.wait(5)
            metadata_failed.set()
            raise ValueError("metadata preparation failed")
        return original_manifest(self, **kwargs)

    monkeypatch.setattr(build.subprocess, "run", fake_run)
    monkeypatch.setattr(type(spec), "to_manifest", fail_manifest)
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(
            build.build_artifact,
            spec,
            "module=__NUMSIM_MODULE__; key=__NUMSIM_CACHE_KEY__",
            cache_dir=tmp_path / "local",
            prepared=prepared,
        )
        try:
            assert metadata_failed.wait(5)
            assert not future.done()
            assert directories[0].is_dir()
        finally:
            finish_compiler.set()
        with pytest.raises(ValueError, match="metadata preparation failed"):
            future.result(timeout=5)
    assert not directories[0].exists()
    assert not prepared.shared_artifact_dir.exists()
