#!/usr/bin/env python3
"""Worker-side prologue bundled by ``kcoral_remote.py`` for kcoral scoring.

The adapter appends ``SOURCES = {...}``: the pinned flashinfer-bench-evolve harness
and one task's modules and definition, verbatim. Inside the worker everything stays in memory —
the harness is imported through an in-memory finder, the workload blobs arrive as
kcoral tensor uploads and are served to the harness's ``load_safetensor``, and the
candidate ``solution.py`` is exec'd as a module. ``main("init", ...)`` runs once per
request; ``main("run")`` scores all workloads through the task's own ``run_suite``.
"""

from __future__ import annotations

import importlib
import importlib.abc
import importlib.machinery
import importlib.util
import json
import linecache
import math
import sys
import types
from dataclasses import replace
from functools import lru_cache

SOURCES: dict[str, str] = {}  # package-relative path -> source or JSON, appended by the adapter
_STATE: dict = {}


class _BundleImporter(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def __init__(self, sources):
        self._modules = {}
        for path, source in sources.items():
            if not path.endswith(".py"):
                continue
            parts = path[: -len(".py")].split("/")
            is_package = parts[-1] == "__init__"
            self._modules[".".join(parts[:-1] if is_package else parts)] = (path, source, is_package)

    def find_spec(self, fullname, path=None, target=None):
        if fullname not in self._modules:
            return None
        path, _, is_package = self._modules[fullname]
        return importlib.machinery.ModuleSpec(fullname, self, origin=f"<bundle:{path}>", is_package=is_package)

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        path, source, _ = self._modules[module.__name__]
        module.__file__ = "/bundle/" + path  # never read; keeps Path(__file__) arithmetic alive
        _exec(source, f"<bundle:{path}>", module.__dict__)


def _exec(source: str, filename: str, namespace: dict) -> None:
    linecache.cache[filename] = (len(source), None, source.splitlines(True), filename)
    # dont_inherit: this file's `from __future__ import annotations` must not leak into
    # the compiled source — TIRx-lite reads live annotation objects at decoration time.
    exec(compile(source, filename, "exec", dont_inherit=True), namespace)


def _sanitize(value):
    """kcoral's JSON refuses NaN/Infinity; carry them as strings."""

    if isinstance(value, float) and not math.isfinite(value):
        return "NaN" if math.isnan(value) else ("Infinity" if value > 0 else "-Infinity")
    if isinstance(value, dict):
        return {key: _sanitize(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize(child) for child in value]
    return value


def _init(task, overrides, workloads, blob_keys, solution_source, *blob_tensors):
    for name in [m for m in sys.modules if m.split(".")[0] == "flashinfer_bench_evolve"]:
        del sys.modules[name]  # a reused worker must not keep an earlier bundle's modules
    sys.meta_path.insert(0, _BundleImporter(SOURCES))
    common = importlib.import_module("flashinfer_bench_evolve.benchmark_common")
    blobs = {(key["path"], key["tensor_key"]): tensor for key, tensor in zip(blob_keys, blob_tensors)}
    common.load_safetensor = lambda spec, device: blobs[(spec["path"], spec["tensor_key"])].to(device).clone()

    @lru_cache(maxsize=None)
    def load_task_reference(task_name, entrypoint="run"):
        # Keep the pinned oracle in memory, just like the uploaded workload blobs.
        path = f"flashinfer_bench_evolve/tasks/{task_name}/definition.json"
        namespace = {}
        _exec(json.loads(SOURCES[path])["reference"], f"<bundle:{path}>", namespace)
        return namespace[entrypoint]

    common.load_task_reference = load_task_reference
    module = importlib.import_module(f"flashinfer_bench_evolve.tasks.{task}.benchmark")  # binds the patched name
    solution = None
    if solution_source is not None:
        solution = sys.modules["solution"] = types.ModuleType("solution")
        _exec(solution_source.decode(), "<solution.py>", solution.__dict__)
    _STATE.update(
        module=module, config=replace(module.default_config(), **overrides), solution=solution, workloads=workloads
    )
    kernels = importlib.util.find_spec("tirx_kernels")
    return {
        "versions": {
            name: importlib.import_module(name).__version__
            for name in ("torch", "tvm", "flashinfer")
            if importlib.util.find_spec(name)
        },
        "tirx_kernels": kernels and kernels.origin,
    }


def _run():
    module, config, solution = _STATE["module"], _STATE["config"], _STATE["solution"]
    workloads = _STATE["workloads"]
    if solution is None:
        rows = module.run_suite(config, workloads=workloads)
    else:
        rows = module.run_suite(
            config,
            candidate_fn=module.tirx_run,
            candidate_prepare_fn=lambda *args: module.tirx_prepare(solution, *args),
            workloads=workloads,
        )
    return _sanitize(rows)


def main(command, *args):
    if command == "init":
        return _init(*args)
    if command == "run":
        return _run(*args)
    raise ValueError(f"unknown driver command {command!r}")
