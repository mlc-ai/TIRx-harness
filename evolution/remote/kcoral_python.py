#!/usr/bin/env python3
"""Run a noninteractive Python script, -c command, or -m module remotely.

Put wrapper options before '--' and Python arguments after it. Upload scripts,
modules, and inputs with --send. Only stdout/stderr and exit status return;
files written remotely are not downloaded.
Use the existing remote environment for kernel diagnostics; do not install
packages, run environment setup, or modify server configuration or shared files.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from evolution.remote.kcoral_tool import (
    add_wrapper_args,
    build_program,
    execute,
    exit_code,
    pack_inputs,
    parse_wrapper_args,
)


def validate_python_args(parser, arguments):
    remaining = iter(arguments)
    for token in remaining:
        if token == "--":
            token = next(remaining, "-")
            if token != "-":
                return
            break
        if token == "-":
            break
        if not token.startswith("-"):
            return
        if token == "--check-hash-based-pycs":
            next(remaining, None)
            continue
        if token.startswith("--"):
            continue
        # Find the target without interpreting its arguments; Python validates other options.
        for index, option in enumerate(token[1:], start=1):
            if option == "i":
                parser.error("interactive Python execution is unsupported")
            if option in "cmWX":
                value = token[index + 1 :] or next(remaining, None)
                if value is None:
                    parser.error(f"Python -{option} requires an argument")
                if option in "cm":
                    return
                break
    parser.error("a script, -c command, or -m module is required; stdin execution is unsupported")


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    add_wrapper_args(parser)
    args, arguments = parse_wrapper_args(parser, argv)
    validate_python_args(parser, arguments)
    return args, arguments


def main(argv=None):
    args, arguments = parse_args(sys.argv[1:] if argv is None else argv)
    try:
        program = build_program(pack_inputs(args.send), [arguments, args.env], function="python")
        return exit_code(execute(args, program)["returncode"])
    except Exception as exc:
        print(f"kcoral_python: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
