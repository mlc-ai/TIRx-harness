"""Render NumSim's declarative TIRx operation support matrix."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from ..abi import NUMSIM_ABI_VERSION
from . import native_frontend

_TILE_SUPPORT_NOTES = {
    "tirx.tile.copy_async": (
        "the engine resolves each TMA shared payload component from concrete runtime addresses "
        "and requires its start to be 128-byte aligned"
    ),
}


def _cell(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def render_engine_support_matrix() -> str:
    """Return the checked-in support matrix for the current NumSim ABI."""

    lines = [
        "# NumSim Engine Operation Support",
        "",
        "This file is generated from NumSim's operation registry and is checked by tests.",
        f"It describes NumSim ABI **v{NUMSIM_ABI_VERSION}**.",
        (
            "A listed operation may still reject modifier, dtype, shape, or layout values "
            "outside its exact specialization domain. Handwritten runtime cases cover "
            "registered operations; GPU parity is tested where the required hardware is available."
        ),
        "",
        "## TIRx CUDA/PTX Ops",
        "",
        "| Operation | Family | Fidelity | Notes |",
        "| --- | --- | --- | --- |",
    ]
    for row in native_frontend.registry_ops():
        if not row["ir_name"].startswith(("tirx.cuda.", "tirx.ptx.")):
            continue
        lines.append(
            "| "
            + " | ".join(
                _cell(value)
                for value in (
                    f"`{row['ir_name']}`",
                    row["family"],
                    row["support"],
                    row["reason"],
                )
            )
            + " |"
        )

    lines.extend(
        (
            "",
            "## CUDA Tile Primitives",
            "",
            (
                "Every warp-, warpgroup-, or CTA-scoped tile call validates complete dynamic "
                "participation in its declared execution scope, including register-only "
                "lowerings with no completion barrier."
            ),
            "",
            "| Operation | Fidelity | Notes |",
            "| --- | --- | --- |",
        )
    )
    for ir_name in native_frontend.registry()["tile_ops"]:
        lines.append(
            "| "
            + " | ".join(
                _cell(value)
                for value in (
                    f"`{ir_name}`",
                    "modeled",
                    _TILE_SUPPORT_NOTES.get(ir_name, ""),
                )
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def write_engine_support_matrix(path: Path | None = None) -> bool:
    """Atomically update the checked-in support matrix when its content changed."""

    if path is None:
        path = Path(__file__).parents[1] / "engine-rs" / "SUPPORTED_OPS.md"
    rendered = render_engine_support_matrix()
    if path.exists() and path.read_text(encoding="utf-8") == rendered:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(rendered, encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return True


def _main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    arguments = parser.parse_args(argv)
    if not arguments.write:
        parser.error("pass --write to update the engine support matrix")
    changed = write_engine_support_matrix()
    print(f"SUPPORTED_OPS.md: {'updated' if changed else 'already current'}")
    return 0


__all__ = ["render_engine_support_matrix", "write_engine_support_matrix"]


if __name__ == "__main__":
    raise SystemExit(_main())
