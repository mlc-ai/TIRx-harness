"""Create a worktree-local venv and remove task-banned kernel sources."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from .guards import CANONICAL_KERNELS_SOURCE_DIR, _sanitize_banned_paths


@dataclass(frozen=True)
class Environment:
    prefix: Path
    python: Path
    site_packages: Path
    kernels: Path

    @classmethod
    def create(cls, worktree: Path) -> Environment:
        """Create a fresh venv and install the worktree's runtime packages."""
        prefix = worktree / ".venv"
        if prefix.exists():
            raise ValueError(f"Run venv already exists: {prefix}")
        python = prefix / "bin/python"
        subprocess.run(
            [
                "uv", "sync", "--locked", "--group", "benchmark", "--no-editable",
                "--link-mode", "copy", "--python", sys.executable,
            ],
            cwd=worktree,
            env={**os.environ, "UV_PROJECT_ENVIRONMENT": str(prefix)},
            check=True,
        )
        # -I ignores PYTHONPATH and the working directory. Discover packages
        # without importing their CUDA/compiler dependencies during preparation.
        result = subprocess.run(
            [
                str(python),
                "-I",
                "-c",
                "import importlib.util, json, sysconfig; "
                "print(json.dumps({'site': sysconfig.get_path('purelib'), "
                "'packages': {n: getattr(importlib.util.find_spec(n), 'origin', None) "
                "for n in ('tirx_kernels', 'tirx_harness')}}))",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        paths = json.loads(result.stdout)
        site = Path(paths["site"]).resolve()
        if not site.is_relative_to(prefix):
            raise ValueError("The venv's site-packages must be inside its own directory.")
        packages = {}
        for name, origin in paths["packages"].items():
            if not origin or not Path(origin).resolve().is_relative_to(site):
                raise ValueError(f"{name} was not installed into the run venv.")
            packages[name] = Path(origin).resolve().parent
        return cls(prefix, python, site, packages["tirx_kernels"])

    def kernel_bans(self, banned_paths: list[str]) -> list[str]:
        """Translate checkout-relative task restrictions to installed paths."""
        # Task declarations retain checkout-relative kernel paths. Translate
        # them to the pip-installed package, without touching another venv.
        package_bans = []
        for pattern in banned_paths:
            normalized = pattern.removeprefix("./")
            if normalized in (
                str(CANONICAL_KERNELS_SOURCE_DIR),
                f"{CANONICAL_KERNELS_SOURCE_DIR}/**",
            ):
                normalized = "tirx_kernels/**"
            else:
                normalized = normalized.removeprefix(f"{CANONICAL_KERNELS_SOURCE_DIR}/")
            if normalized == "tirx_kernels" or normalized.startswith("tirx_kernels/"):
                package_bans.append(str(self.site_packages / normalized))
        return package_bans

    def sanitize(self, banned_paths: list[str]) -> None:
        """Remove task-banned kernel sources and the wheel's skill copies from this venv."""
        _sanitize_banned_paths(self.site_packages, self.kernel_bans(banned_paths))
        # Runs install skills from the worktree; leave no unsanitized second copy.
        shutil.rmtree(self.site_packages / "tirx_harness" / "_skills", ignore_errors=True)
