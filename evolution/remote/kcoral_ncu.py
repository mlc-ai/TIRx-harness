#!/usr/bin/env python3
"""Capture NCU remotely and save its report to the required local -o path.

Put wrapper and NCU options before '--', then the application and its arguments.
Upload scripts and inputs with --send. Inspect returned reports with local ncu.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from evolution.remote.kcoral_tool import (
    add_wrapper_args,
    build_program,
    execute,
    exit_code,
    pack_inputs,
    save_file,
    validate_wrapper_args,
)


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description=__doc__,
        allow_abbrev=False,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_wrapper_args(parser)
    parser.add_argument(
        "-o",
        "--export",
        required=True,
        type=Path,
        help="exact LOCAL report filename (required; no extension is added)",
    )
    parser.add_argument(
        "-f", "--force-overwrite", action="store_true", help="replace an existing local report"
    )
    if "--" not in argv:
        if "--help" in argv or "-h" in argv:
            parser.parse_args(["--help"])
        parser.error("separate the application from capture options with '--'")
    boundary = argv.index("--")
    args, ncu_args = parser.parse_known_args(argv[:boundary])
    application = argv[boundary + 1 :]
    if not application:
        parser.error("an application is required after '--'")
    validate_wrapper_args(parser, args)
    # Capture only: report import and attach/launch-only belong outside this tool.
    for token in ncu_args:
        option = token.split("=", 1)[0]
        if option in {"-i", "--import", "--mode", "--config-file", "--config-file-path"} or (
            token.startswith("-i") and not token.startswith("--")
        ):
            parser.error(f"{option} is not supported for remote capture; inspect reports locally")
        if option in {"-h", "--help", "-v", "--version"}:
            parser.error("use local ncu for help/version queries")
    return args, ncu_args, application


def main(argv=None):
    args, ncu_args, application = parse_args(sys.argv[1:] if argv is None else argv)
    try:
        output = args.export.absolute()
        if output.is_dir():
            raise ValueError(f"-o must name a local file, not a directory: {output}")
        if os.path.lexists(output) and not args.force_overwrite:
            raise ValueError(f"output already exists (use -f to replace): {output}")
        archive = pack_inputs(args.send)
        output.parent.mkdir(parents=True, exist_ok=True)
        program = build_program(archive, [ncu_args, application, args.env], function="ncu")
        capture = execute(args, program)
        code = capture["returncode"]
        if capture["report"] is not None:
            save_file(output, capture["report"], args.force_overwrite)
            print(f"kcoral_ncu: report saved to {output}", file=sys.stderr)
        else:
            print("kcoral_ncu: NCU produced no report", file=sys.stderr)
            if code == 0:
                code = 1
        return exit_code(code)
    except Exception as exc:
        print(f"kcoral_ncu: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
