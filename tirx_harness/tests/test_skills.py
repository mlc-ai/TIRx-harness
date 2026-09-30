"""The bundled skills install into an agent's skills directory."""

import pytest

from tirx_harness import cli, skills


@pytest.fixture
def bundle(tmp_path, monkeypatch):
    """A bundled-skills tree whose wiki fetcher records where it ran."""
    root = tmp_path / "bundle"
    for name in ("tirx-debug-kernel", "tirx-wiki"):
        (root / name).mkdir(parents=True)
        (root / name / "SKILL.md").write_text(f"# {name}\n")
    fetcher = root / skills.REFERENCE_FETCHER
    fetcher.parent.mkdir()
    fetcher.write_text(
        "from pathlib import Path\n"
        "(Path(__file__).resolve().parents[1] / 'fetched').write_text('ok')\n"
    )
    monkeypatch.setattr(skills, "BUNDLED_SKILLS", root)
    return root


def test_install_copies_skills_and_fetches_into_the_copy(bundle, tmp_path):
    dest = tmp_path / "project" / ".agents" / "skills"

    installed = skills.install_skills(dest)

    assert installed == [dest / "tirx-debug-kernel", dest / "tirx-wiki"]
    assert (dest / "tirx-debug-kernel" / "SKILL.md").read_text() == "# tirx-debug-kernel\n"
    assert (dest / "tirx-wiki" / "fetched").read_text() == "ok"
    assert not (bundle / "tirx-wiki" / "fetched").exists()


def test_install_replaces_existing_skills_only_with_force(bundle, tmp_path):
    (tmp_path / "tirx-wiki").mkdir()
    (tmp_path / "tirx-wiki" / "stale").write_text("old")
    (tmp_path / "other-skill").mkdir()

    with pytest.raises(FileExistsError):
        skills.install_skills(tmp_path, fetch=False)
    assert not (tmp_path / "tirx-debug-kernel").exists()

    skills.install_skills(tmp_path, fetch=False, force=True)
    assert not (tmp_path / "tirx-wiki" / "stale").exists()
    assert (tmp_path / "tirx-wiki" / "SKILL.md").is_file()
    assert (tmp_path / "other-skill").is_dir()


def test_cli_lists_and_installs(bundle, tmp_path, capsys):
    assert cli.main(["skills", "list"]) == 0
    assert capsys.readouterr().out.splitlines() == ["tirx-debug-kernel", "tirx-wiki"]

    dest = tmp_path / "skills"
    assert cli.main(["skills", "install", "--dest", str(dest), "--no-fetch"]) == 0
    assert (dest / "tirx-wiki" / "SKILL.md").is_file()

    assert cli.main(["skills", "install", "--dest", str(dest), "--no-fetch"]) == 1


def test_cli_reports_a_failed_fetch(bundle, tmp_path):
    (bundle / skills.REFERENCE_FETCHER).write_text("raise SystemExit(3)\n")

    assert cli.main(["skills", "install", "--dest", str(tmp_path)]) == 1

    assert (tmp_path / "tirx-wiki" / "SKILL.md").is_file()
