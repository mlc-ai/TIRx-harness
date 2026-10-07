"""The ``tirx-harness`` command."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import skills


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tirx-harness")
    commands = parser.add_subparsers(dest="command", required=True)
    skill_parser = commands.add_parser("skills", help="manage the bundled agent skills")
    skill_commands = skill_parser.add_subparsers(dest="skills_command", required=True)
    skill_commands.add_parser("list", help="list the bundled skills")
    install = skill_commands.add_parser(
        "install", help="copy all skills into an agent's skills directory"
    )
    install.add_argument(
        "--dest",
        type=Path,
        required=True,
        help="the agent's skills directory, such as .agents/skills or .claude/skills",
    )
    install.add_argument(
        "--no-fetch", action="store_true", help="skip downloading the tirx-wiki references"
    )
    install.add_argument(
        "--force", action="store_true", help="replace skill directories that already exist"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.skills_command == "list":
            print("\n".join(skills.available_skills()))
            return 0
        installed = skills.install_skills(args.dest, fetch=not args.no_fetch, force=args.force)
    except (FileExistsError, FileNotFoundError, skills.ReferenceFetchError) as error:
        print(f"tirx-harness: error: {error}", file=sys.stderr)
        return 1
    for path in installed:
        print(f"installed {path}")
    return 0
