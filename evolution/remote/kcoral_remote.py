#!/usr/bin/env python3
"""Score candidates or run kernel diagnostics on an existing kcoral benchmark server.

    python evolution/remote/kcoral_remote.py <workload> [version] --remote URL                  # score vN (or the baseline)
    python evolution/remote/kcoral_remote.py --remote URL --cmd 'nvidia-smi'                    # inspect GPU status

Use the server's existing tools and dependencies. Do not install, upgrade, or
remove packages, run setup scripts, edit server configuration, or manage server
processes through this tool. Report missing dependencies instead of changing
the remote environment. These are usage constraints, not a sandbox enforced
by the command runner.

``--cmd`` short-circuits everything below: it uploads one small module and has the worker
``subprocess.run(["bash", "-c", script])``. Nothing else travels and no workload is needed;
the command's stdout/stderr are the request's own capture, because the worker redirects
fd 1 and 2 and a subprocess writes straight into them. It runs on the server machine as
the server's user, with the worker's ``CUDA_VISIBLE_DEVICES`` and access to that
machine's filesystem.

``--send PATH`` adds one uncompressed tar of PATH as a ``bytes`` upload (gzip costs more
than the transfer it saves at the server's parse speed). The worker unpacks it into a
fresh temp directory, runs the command there with ``$KCORAL_DIR`` pointing at it, and
removes it afterwards. Keep explicit scratch and output writes in that directory;
do not copy files into persistent server locations. Preserve results locally via
captured stdout/stderr or the dedicated tool CLIs' artifact-return options.

Without ``--cmd``, a request carries everything the worker needs: ONE Python module
assembled locally (``kcoral_driver.py`` plus the pinned harness and task sources,
verbatim), the safetensors blobs it references as tensor uploads, the candidate
``solution.py`` as bytes, then ``init`` on the worker.

**Every benchmark re-ships all of that.** One request carries the full workload list
and calls ``init`` and ``run`` once. All shapes share a worker and its in-process
compilation caches; suite-wide correctness gates run once. kcoral may be a fleet,
so each scoring run remains self-contained and does not depend on earlier requests.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from evolution.benchmark.adapter import (
    CANDIDATES_ROOT,
    PACKAGED,
    blobs,
    harness_sources,
    plan,
    workload_key,
)

DRIVER_PATH = Path(__file__).with_name("kcoral_driver.py")


def _bundle(task_name: str) -> str:
    """The one module kcoral runs: ``kcoral_driver.py`` plus the adapter's harness sources."""

    return DRIVER_PATH.read_text() + "\nSOURCES = " + json.dumps(harness_sources(task_name)) + "\n"


def _request(task_name: str, overrides: dict, workloads: list, candidate: bytes | None):
    """One self-contained request: the bundle, these workloads' blobs, the candidate and ``init``.

    Nothing here counts on the server remembering an earlier request. kcoral may be a
    fleet, and the machine that served the last benchmark may be gone or may never see
    this one, so every request re-ships everything the worker needs.
    """

    from kcoral import Program

    keys, tensors = blobs(workloads)
    program = Program()
    module = program.upload(id="bundle", kind="module", source=_bundle(task_name))
    bundle = program.get_function(id="main", module=module, name="main")
    solution = (
        None if candidate is None else program.upload(id="solution", kind="bytes", value=candidate)
    )
    handles = [
        program.upload(id=f"blob{i}", kind="tensor", value=tensor)
        for i, tensor in enumerate(tensors)
    ]
    init = program.run(
        id="init",
        fn=bundle,
        args=["init", task_name, overrides, workloads, keys, solution, *handles],
    )
    program.return_(key="init", value=init)
    return program, bundle, handles


def _execute(url: str, program, timeout_seconds: int, output_limit_bytes: int | None = None):
    """Send ``program``, replay the worker's output, fail loudly on a FAILED request."""

    from kcoral import Client

    with Client(url) as client:
        result = client.execute(
            program, timeout_seconds=timeout_seconds, output_limit_bytes=output_limit_bytes
        )
    sys.stdout.write(result.stdout)  # the harness's own output, verbatim
    sys.stderr.write(result.stderr)
    if not result.completed:
        error = result.error or {}
        raise SystemExit(
            f"benchmark server request {result.status} ({error.get('kind')} at "
            f"{error.get('instruction_id')}): {error.get('message')}\n{error.get('traceback', '')}"
        )
    return result


def run_remote(
    task_name: str,
    workload_dir: Path,
    version: str,
    *,
    warmup: int,
    repeat: int,
    shape_mode: str,
    url: str,
    timeout: int,
):
    """Score ``vN`` (or the baseline): one self-contained request for the full suite."""

    overrides, workloads, candidate = plan(
        task_name, workload_dir, version, warmup=warmup, repeat=repeat, shape_mode=shape_mode
    )
    program, bundle, handles = _request(task_name, overrides, workloads, candidate)
    program.return_(key="rows", value=program.run(id="run", fn=bundle, args=["run"]))
    print(f"benchmark server: {url} ({len(workloads)} workloads, {len(handles)} blob tensors)")
    result = _execute(url, program, timeout)
    print(f"worker: {result.results['init']}")
    return result.results["rows"]


RUNNER = '''\
import io
import os
import shutil
import subprocess
import tarfile
import tempfile
import time


def main(script, archive=None):
    """One bash script in the worker; its stdout/stderr are the request's own capture."""

    start = time.monotonic()
    workdir, env = None, dict(os.environ)
    if archive is not None:
        workdir = tempfile.mkdtemp(prefix="kcoral_send_")
        with tarfile.open(fileobj=io.BytesIO(archive), mode="r:") as tar:
            try:
                tar.extractall(workdir, filter="data")  # refuses paths outside workdir
            except TypeError:  # a Python without the extraction filters
                tar.extractall(workdir)
        env["KCORAL_DIR"] = workdir
    try:
        completed = subprocess.run(["bash", "-c", script], cwd=workdir, env=env)
    finally:
        if workdir is not None:
            shutil.rmtree(workdir, ignore_errors=True)
    return {"returncode": completed.returncode, "seconds": round(time.monotonic() - start, 3)}
'''


def _tar(path: Path) -> bytes:
    """One uncompressed tar of ``path``: a directory's contents, or a single file."""

    import tarfile

    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as tar:
        if path.is_dir():
            for child in sorted(path.rglob("*")):
                if child.is_file() and "__pycache__" not in child.parts:
                    tar.add(child, arcname=str(child.relative_to(path)))
        else:
            tar.add(path, arcname=path.name)
    return buffer.getvalue()


OUTPUT_LIMIT_BYTES = 16 * 1024**2  # the server's maximum; a build log outgrows the 1 MB default


def run_cmd(script: str, send: Path | None = None, *, url: str, timeout: int) -> int:
    """Run ``script`` under bash on the server: one module upload, one run, no workload."""

    from kcoral import Program

    program = Program()
    module = program.upload(id="runner", kind="module", source=RUNNER)
    runner = program.get_function(id="main", module=module, name="main")
    args: list = [script]
    sent = ""
    if send is not None:
        archive = _tar(send)
        args.append(program.upload(id="send", kind="bytes", value=archive))
        sent = f", {send} as {len(archive) / 1024**2:.2f} MiB tar"
    program.return_(key="cmd", value=program.run(id="cmd", fn=runner, args=args))
    print(f"benchmark server: {url} (bash, {len(script)} chars{sent})")
    result = _execute(url, program, timeout, OUTPUT_LIMIT_BYTES)
    if result.stdout_truncated or result.stderr_truncated:
        print(f"warning: output truncated at {OUTPUT_LIMIT_BYTES} bytes", file=sys.stderr)
    outcome = result.results["cmd"]
    print(f"exit {outcome['returncode']} after {outcome['seconds']}s")
    return outcome["returncode"]


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "workload", nargs="?", default=None, help=f"packaged workload: {', '.join(PACKAGED)}"
    )
    parser.add_argument(
        "version",
        nargs="?",
        default=None,
        help="Version directory (e.g. v0) or 'baseline' (default).",
    )
    parser.add_argument(
        "--remote",
        metavar="URL",
        required=True,
        help="the kcoral benchmark server (must already be running)",
    )
    parser.add_argument("--warmup", type=int, default=None, help="default: per-workload value")
    parser.add_argument("--repeat", type=int, default=None, help="default: per-workload value")
    parser.add_argument(
        "--cmd",
        metavar="CMD",
        default=None,
        help="run a kernel diagnostic CMD under bash (or a diagnostic bash file); no remote environment changes",
    )
    parser.add_argument(
        "--send",
        metavar="PATH",
        type=Path,
        default=None,
        help="with --cmd: ship PATH as a tar, unpacked into the command's cwd ($KCORAL_DIR)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=3600,
        help="server-side seconds for the full request, including all workloads (max 3600)",
    )
    args = parser.parse_args(argv)
    if args.send is not None and args.cmd is None:
        parser.error("--send needs --cmd: nothing would run the unpacked files")
    if args.cmd is not None:
        source = Path(args.cmd)
        script = (
            source.read_text() if os.path.isfile(source) else args.cmd
        )  # a long inline script is no path
        if args.send is not None and not args.send.exists():
            parser.error(f"--send path does not exist: {args.send}")
        raise SystemExit(run_cmd(script, args.send, url=args.remote, timeout=args.timeout))
    if args.workload is None:
        parser.error("a workload is required unless --cmd is given")
    key = workload_key(args.workload)
    if key not in PACKAGED:
        parser.error(f"only packaged workloads run on the server, not {key!r}")
    task, warmup, repeat, shape_mode = PACKAGED[key]
    kwargs = dict(
        warmup=warmup if args.warmup is None else args.warmup,
        repeat=repeat if args.repeat is None else args.repeat,
        shape_mode=shape_mode,
        url=args.remote,
        timeout=args.timeout,
    )
    version = "baseline" if args.version is None else args.version
    run_remote(task, CANDIDATES_ROOT / key, version, **kwargs)


if __name__ == "__main__":
    main()
