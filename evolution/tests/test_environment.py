"""Run venvs keep no second copy of the skills installed in the worktree."""

from evolution.preparation.environment import Environment


def test_sanitize_removes_the_wheel_skill_copies(tmp_path):
    site = tmp_path / "venv" / "site-packages"
    bundled = site / "tirx_harness" / "_skills" / "tirx-wiki"
    bundled.mkdir(parents=True)
    (bundled / "SKILL.md").write_text("# tirx-wiki\n")
    (site / "tirx_harness" / "__init__.py").write_text("")
    kernels = site / "tirx_kernels"
    kernels.mkdir()
    environment = Environment(tmp_path / "venv", tmp_path / "venv/bin/python", site, kernels)

    environment.sanitize([])

    assert not (site / "tirx_harness" / "_skills").exists()
    assert (site / "tirx_harness" / "__init__.py").is_file()
    assert kernels.is_dir()
