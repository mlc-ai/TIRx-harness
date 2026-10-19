#!/usr/bin/env python3
"""Run Compute Sanitizer remotely, with optional log/record retrieval.

Put wrapper options before '--', followed by native sanitizer/application args.
Use --error-exitcode 1 to make sanitizer findings fail the command.
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
    relative_path,
    save_artifacts,
)


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    add_wrapper_args(parser)
    parser.add_argument(
        "--fetch",
        action="append",
        default=[],
        help="remote relative log/record file or directory to return; repeatable",
    )
    parser.add_argument("--output-dir", type=Path, help="local directory for fetched artifacts")
    parser.add_argument("-f", "--force-overwrite", action="store_true")
    args, forwarded = parse_wrapper_args(parser, argv)
    if bool(args.fetch) != bool(args.output_dir):
        parser.error("--fetch and --output-dir must be used together")
    for name in args.fetch:
        try:
            relative_path(name)
        except ValueError as exc:
            parser.error(str(exc))
    return args, forwarded


def main(argv=None):
    args, forwarded = parse_args(sys.argv[1:] if argv is None else argv)
    try:
        prepare_output_dir(args.output_dir)
        program = build_program(
            pack_inputs(args.send), [forwarded, args.fetch, args.env], function="compute_sanitizer"
        )
        outcome = execute(args, program)
        save_artifacts(args.output_dir, outcome["files"], overwrite=args.force_overwrite)
        return artifact_exit_code(outcome)
    except Exception as exc:
        print(f"kcoral_compute_sanitizer: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
