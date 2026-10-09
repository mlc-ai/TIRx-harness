"""Setup fails when a kernel ban no longer matches the fetched kernel checkout."""

import pytest

from evolution.preparation import live_references as refs


def test_reference_fetch_rejects_a_kernel_ban_that_matches_nothing(tmp_path):
    (tmp_path / refs.REPOSITORY_ROOT_REL / "tirx-kernels/tirx_kernels/kda").mkdir(parents=True)
    (tmp_path / refs.CATALOG_REL).write_text('{"manuals": {}, "repositories": {}}')
    (tmp_path / refs.FETCH_SCRIPT_REL).parent.mkdir()
    (tmp_path / refs.FETCH_SCRIPT_REL).write_text("")
    (tmp_path / ".agents/skills").mkdir(parents=True)
    (tmp_path / ".agents/skills/tirx-wiki").symlink_to(tmp_path / refs.SKILL_REL)

    refs.fetch_live_references(tmp_path, ["tirx-kernels/tirx_kernels/kda/**"])
    with pytest.raises(ValueError, match="tirx_kernels/flashinfer/kda"):
        refs.fetch_live_references(tmp_path, ["tirx-kernels/tirx_kernels/flashinfer/kda/**"])
