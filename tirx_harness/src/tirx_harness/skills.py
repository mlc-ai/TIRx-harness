"""Install the agent skills bundled with tirx-harness."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

BUNDLED_SKILLS = Path(__file__).resolve().parent / "_skills"
REFERENCE_FETCHER = Path("tirx-wiki") / "scripts" / "fetch_references.py"

__all__ = [
    "BUNDLED_SKILLS",
    "REFERENCE_FETCHER",
    "ReferenceFetchError",
    "available_skills",
    "install_skills",
]


class ReferenceFetchError(RuntimeError):
    """The skills were copied, but fetching the tirx-wiki references failed."""


def available_skills() -> dict[str, Path]:
    """Map each bundled skill name to its source directory."""
    if not BUNDLED_SKILLS.is_dir():
        raise FileNotFoundError(
            f"no bundled skills at {BUNDLED_SKILLS}; editable installs do not bundle them, "
            "so copy them from the repository's skills/ directory instead"
        )
    return {path.parent.name: path.parent for path in sorted(BUNDLED_SKILLS.glob("*/SKILL.md"))}


def install_skills(dest: Path, *, fetch: bool = True, force: bool = False) -> list[Path]:
    """Copy every bundled skill into ``dest`` and fetch the wiki references into the copy.

    The skills work together, so they install as a set. The fetcher writes
    beside itself, so it runs from the installed copy rather than from this package.
    """
    skills = available_skills()
    dest = Path(dest)
    targets = [dest / name for name in skills]
    existing = [target for target in targets if target.exists() or target.is_symlink()]
    if existing and not force:
        raise FileExistsError(
            f"{', '.join(map(str, existing))} already exist; use --force to replace them"
        )

    dest.mkdir(parents=True, exist_ok=True)
    for source, target in zip(skills.values(), targets):
        if target.is_symlink() or target.is_file():
            target.unlink()
        elif target.exists():
            shutil.rmtree(target)
        shutil.copytree(source, target, ignore=shutil.ignore_patterns("__pycache__"))

    if fetch:
        command = [sys.executable, str(dest / REFERENCE_FETCHER)]
        try:
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError as error:
            raise ReferenceFetchError(
                "skills were installed, but fetching the tirx-wiki references failed; "
                f"rerun: {' '.join(command)}"
            ) from error
    return targets
