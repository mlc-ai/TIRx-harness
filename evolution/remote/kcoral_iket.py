#!/usr/bin/env python3
"""Profile remotely with IKET and save the output directory locally.

Put wrapper options before '--', followed by run-iket options and
'profile [options] -- application [args]'. Inspect returned traces locally.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from evolution.remote.kcoral_tool import (
    add_wrapper_args,
    artifact_exit_code,
    build_program,
    execute,
    pack_inputs,
    parse_wrapper_args,
    prepare_output_dir,
    save_artifacts,
)


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    add_wrapper_args(parser)
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="local directory for IKET traces and retained intermediates",
    )
    parser.add_argument("-f", "--force-overwrite", action="store_true")
    args, forwarded = parse_wrapper_args(parser, argv)
    if "--" not in forwarded:
        parser.error("separate IKET profile options from the application with another '--'")
    boundary = forwarded.index("--")
    options = forwarded[:boundary]
    if "profile" not in options or not forwarded[boundary + 1 :]:
        parser.error("expected 'profile [options] -- application [args]'")
    for token in options:
        option = token.split("=", 1)[0]
        if (option.startswith("-o") and not option.startswith("--")) or (
            option.startswith("--")
            and any(managed.startswith(option) for managed in ("--output-dir", "--working-dir"))
        ):
            parser.error("remote output and working directories are managed by the wrapper")
    return args, forwarded


def main(argv=None):
    args, forwarded = parse_args(sys.argv[1:] if argv is None else argv)
    try:
        prepare_output_dir(args.output_dir)
        program = build_program(pack_inputs(args.send), [forwarded, args.env], function="iket")
        outcome = execute(args, program)
        save_artifacts(args.output_dir, outcome["files"], overwrite=args.force_overwrite)
        return artifact_exit_code(outcome)
    except Exception as exc:
        print(f"kcoral_iket: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
