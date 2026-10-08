from __future__ import annotations

import numpy as np

from tirx_harness import numsim
from tests.numsim.support.kernels import lane_add
from tirx_harness.numsim.transpiler import build


def test_profile_feature_is_opt_in_and_changes_generated_cargo(monkeypatch, tmp_path):
    lib_target = build._cargo_lib_target("ab" * 32)
    monkeypatch.delenv("NUMSIM_PROFILE", raising=False)
    plain = build._cargo_toml(tmp_path, lib_target)
    assert 'features = ["python"]' in plain

    monkeypatch.setenv("NUMSIM_PROFILE", "1")
    profiled = build._cargo_toml(tmp_path, lib_target)
    assert 'features = ["python", "profile"]' in profiled


def test_generated_cargo_uses_a_stable_package_and_artifact_specific_library_target(tmp_path):
    first_target = build._cargo_lib_target("ab" * 32)
    second_target = build._cargo_lib_target("cd" * 32)
    manifest = build._cargo_toml(tmp_path, first_target)

    assert f'name = "{build._CARGO_PACKAGE_NAME}"' in manifest
    assert f'name = "{first_target}"' in manifest
    assert f'[profile.release.package."{build._CARGO_PACKAGE_NAME}"]' in manifest
    assert f"opt-level = {build._GENERATED_OPT_LEVEL}" in manifest
    assert f"codegen-units = {build._GENERATED_CODEGEN_UNITS}" in manifest
    assert f'[profile.release]\nlto = "{build._GENERATED_ARTIFACT_RUSTC_LTO}"' in manifest
    assert first_target != second_target


def test_generated_cargo_keeps_racecheck_out_of_synccheck_dependency_metadata(
    monkeypatch, tmp_path
):
    lib_target = build._cargo_lib_target("ab" * 32)
    monkeypatch.delenv("NUMSIM_PROFILE", raising=False)

    synccheck = build._cargo_toml(
        tmp_path,
        lib_target,
        analysis_capable=True,
        analysis_checker="synccheck",
    )
    racecheck = build._cargo_toml(
        tmp_path,
        lib_target,
        analysis_capable=True,
        analysis_checker="racecheck",
    )

    assert 'features = ["python", "analysis-core"]' in synccheck
    assert 'features = ["python", "analysis"]' in racecheck


def test_profile_feature_is_an_observable_artifact_stats_surface(
    monkeypatch, tmp_path, expect_harness_surface
):
    monkeypatch.setenv("NUMSIM_PROFILE", "1")
    module = numsim.transpile(lane_add, cache_dir=tmp_path)
    result = numsim.Engine(max_workers=1).run(
        module,
        {
            "left": np.arange(100, dtype=np.float32),
            "right": np.ones(100, dtype=np.float32),
            "output": np.zeros(100, dtype=np.float32),
        },
    )

    profile = result.stats["kernels"][0]["profile"]
    assert set(profile) >= {"worker_total", "future_poll", "gmem_read", "gmem_write"}
    assert profile["worker_total"]["count"] >= 1
    assert profile["future_poll"]["count"] >= result.stats["task_count"]
    # Scalar warp I/O records one profiled batch per static memory operation,
    # rather than one timer entry per active lane.
    assert profile["gmem_read"]["count"] == 8
    assert profile["gmem_write"]["count"] == 4

    def check_profile(value):
        assert value["worker_total"]["count"] >= 1
        assert value["future_poll"]["count"] >= result.stats["task_count"]

    expect_harness_surface(lambda: profile, check_profile)
