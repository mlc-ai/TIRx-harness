"""Shared CPU-only transport for the dedicated KCoral tool CLIs."""

from __future__ import annotations

import io
import os
import re
import sys
import tarfile
import tempfile
from pathlib import Path, PurePosixPath

OUTPUT_LIMIT_BYTES = 16 * 1024**2
WORKER_PATH = Path(__file__).with_name("kcoral_tool_worker.py")


def relative_path(name):
    path = PurePosixPath(name)
    if not name or path.is_absolute() or ".." in path.parts or "\\" in name or not path.parts:
        raise ValueError(f"expected a relative file path: {name!r}")
    return path


def pack_inputs(paths=()):
    """Stable archives: directory contents and file basenames."""
    buffer, names = io.BytesIO(), set()
    with tarfile.open(fileobj=buffer, mode="w") as tar:

        def add(name, data, executable=False):
            name = relative_path(name).as_posix()
            if name in names or any(
                name.startswith(old + "/") or old.startswith(name + "/") for old in names
            ):
                raise ValueError(f"duplicate or conflicting input path: {name}")
            names.add(name)
            info = tarfile.TarInfo(name)
            info.size = len(data)
            info.mode = 0o700 if executable else 0o600
            tar.addfile(info, io.BytesIO(data))

        for path in paths:
            if path.is_symlink() or not path.exists():
                raise ValueError(f"input must exist and not be a symlink: {path}")
            for entry in sorted(path.rglob("*")) if path.is_dir() else [path]:
                if "__pycache__" in entry.parts:
                    continue
                if entry.is_symlink():
                    raise ValueError(f"symlink inputs are not supported: {entry}")
                if entry.is_dir():
                    continue
                if not entry.is_file():
                    raise ValueError(f"input is not a regular file: {entry}")
                name = entry.relative_to(path).as_posix() if path.is_dir() else entry.name
                add(name, entry.read_bytes(), bool(entry.stat().st_mode & 0o111))
    return buffer.getvalue()


def add_wrapper_args(parser):
    parser.add_argument("--remote", required=True, help="KCoral server URL")
    parser.add_argument(
        "--send",
        type=Path,
        action="append",
        default=[],
        help="upload a file or directory's contents; repeatable",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="request timeout, 1-3600s (default: 300); increase only if needed",
    )
    parser.add_argument(
        "-e",
        "--env",
        action="append",
        default=[],
        metavar="NAME[=VALUE]",
        help=(
            "set a variable for this request's subprocess; "
            "NAME alone copies its local value (repeatable)"
        ),
    )


def validate_wrapper_args(parser, args):
    if not 1 <= args.timeout <= 3600:
        parser.error("--timeout must be between 1 and 3600")
    environment = {}
    for entry in args.env:
        name, separator, value = entry.partition("=")
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
            parser.error("environment variable names must be identifiers")
        if name in {"CUDA_VISIBLE_DEVICES", "KCORAL_DIR"}:
            parser.error(f"{name} is managed by the remote worker")
        if not separator:
            if name not in os.environ:
                parser.error(f"local environment variable {name} is not set")
            value = os.environ[name]
        if "\x00" in value:
            parser.error(f"environment variable {name} contains NUL")
        environment[name] = value
    args.env = environment


def parse_wrapper_args(parser, argv):
    if "--" not in argv:
        if "--help" in argv or "-h" in argv:
            parser.parse_args(["--help"])
        parser.error("separate wrapper options from tool arguments with '--'")
    boundary = argv.index("--")
    args = parser.parse_args(argv[:boundary])
    forwarded = argv[boundary + 1 :]
    if not forwarded:
        parser.error("arguments are required after '--'")
    validate_wrapper_args(parser, args)
    return args, forwarded


def build_program(archive, arguments, *, function):
    from kcoral import Program

    program = Program()
    module = program.upload(id="tool_runner", kind="module", source=WORKER_PATH.read_text())
    runner = program.get_function(id="runner", module=module, name=function)
    inputs = program.upload(id="inputs", kind="bytes", value=archive)
    outcome = program.run(id="run", fn=runner, args=[inputs, *arguments])
    program.return_(key="outcome", value=outcome)
    return program


def execute(args, program):
    from kcoral import Client

    with Client(args.remote) as client:
        result = client.execute(
            program, timeout_seconds=args.timeout, output_limit_bytes=OUTPUT_LIMIT_BYTES
        )
    sys.stdout.write(result.stdout)
    sys.stderr.write(result.stderr)
    if result.stdout_truncated or result.stderr_truncated:
        print("kcoral: remote output was truncated", file=sys.stderr)
    if not result.completed:
        error = result.error or {}
        raise RuntimeError(
            f"{result.status}: {error.get('message', error)}\n{error.get('traceback', '')}"
        )
    return result.results["outcome"]


def exit_code(code):
    return code if code >= 0 else 128 - code


def save_file(path, data, overwrite=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".kcoral-", delete=False) as file:
        temporary = Path(file.name)
        try:
            file.write(data)
            file.close()
            if overwrite:
                os.replace(temporary, path)
            else:
                os.link(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


def prepare_output_dir(path):
    if path is not None:
        path.mkdir(parents=True, exist_ok=True)


def save_artifacts(output_dir, files, *, overwrite):
    base = output_dir.absolute() if output_dir else None
    for name, data in files.items():
        output = base.joinpath(*relative_path(name).parts)
        if not output.parent.resolve().is_relative_to(base.resolve()):
            raise ValueError(f"artifact parent escapes local output directory: {output}")
        save_file(output, data, overwrite)
        print(f"kcoral: saved {output}", file=sys.stderr)


def artifact_exit_code(outcome):
    code = exit_code(outcome["returncode"])
    if outcome["missing"]:
        print(f"kcoral: missing artifacts: {', '.join(outcome['missing'])}", file=sys.stderr)
        code = code or 1
    return code
