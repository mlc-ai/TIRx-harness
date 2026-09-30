"""Dump CUDA / PTX / SASS from a compiled TIRx module (the ``tvm.compile`` output).

In-process API that turns a compiled module into its generated CUDA source and,
via nvcc / ptxas / cuobjdump, PTX / cubin / SASS plus the ``ptxas -v``
register/spill/SMEM report — so you don't run that pipeline by hand::

    from tirx_harness.dump_kernel import dump_module, dump_cuda

    ex = tvm.compile(tvm.IRModule({"main": pf}), target="cuda", tir_pipeline="tirx")
    d = dump_module(ex, ptx=True)      # -> DumpResult
    print(d.cuda, d.ptx, d.sass)       # artifact text
    print(d.ptxas["registers"])        # parsed ptxas -v report
    print(d.symbols)                   # ['_kernel_kernel']

    dump_module(ex, outdir="/tmp/dumps", name="v0")   # also persist files
    cuda = dump_cuda(ex)               # just the CUDA source string

Both accept the ``Executable`` from ``tvm.compile`` or a runtime ``Module``.
Arch is auto-detected from the local GPU (B200 -> ``sm_100a``); the
architecture-specific ``a`` suffix (needed for ``tcgen05.*`` on Blackwell and
Hopper-only opcodes) is added automatically for SM 9.x/10.x.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

CUDA_HOME = os.environ.get("CUDA_HOME", "/usr/local/cuda")
DEFAULT_ARCH = "sm_100a"  # fallback when GPU probe fails (this box is B200)


def _cuda_tool(name: str) -> str:
    """Resolve a CUDA executable: ``$CUDA_HOME/bin``, then ``PATH``, then the default."""
    cand = Path(CUDA_HOME) / "bin" / name
    if cand.exists():
        return str(cand)
    return shutil.which(name) or str(cand)


def _cuda_include() -> str:
    """Resolve the CUDA include dir: ``$CUDA_HOME/include``, else derived from nvcc on PATH."""
    inc = Path(CUDA_HOME) / "include"
    if inc.exists():
        return str(inc)
    nvcc = shutil.which("nvcc")
    if nvcc:
        derived = Path(nvcc).resolve().parent.parent / "include"
        if derived.exists():
            return str(derived)
    return str(inc)


# ============================================================================
# Arch detection
# ============================================================================


def detect_arch() -> str:
    """Return the nvcc/ptxas ``-arch`` string for the local GPU.

    ``nvidia-smi`` reports the compute capability (e.g. ``10.0`` for B200); we
    map ``<major>.<minor>`` to ``sm_<major><minor>`` and append the ``a``
    architecture-specific suffix for SM >= 9.0 (Hopper/Blackwell), which is
    required for ``tcgen05.*`` / ``cp.async.bulk.tensor`` / other family-only
    opcodes. Falls back to :data:`DEFAULT_ARCH` if the probe fails.
    """
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        cap = out.stdout.strip().splitlines()[0].strip()
        major, minor = (int(x) for x in cap.split("."))
        suffix = "a" if major >= 9 else ""
        return f"sm_{major}{minor}{suffix}"
    except Exception:
        return DEFAULT_ARCH


# ============================================================================
# Compiled module -> CUDA source
# ============================================================================


def _iter_modules(m):
    """Yield a runtime module and all of its (recursive) imported submodules."""
    yield m
    try:
        imports = m.imports
    except Exception:
        imports = getattr(m, "imports_", [])
    for im in imports:
        yield from _iter_modules(im)


def dump_cuda(mod) -> str:
    """Extract generated CUDA source from a compiled module (the ``tvm.compile`` output).

    Accepts the ``Executable`` returned by ``tvm.compile`` (uses its ``.mod``) or a
    runtime ``Module`` directly. The device CUDA lives in an *imported* submodule
    (the top-level host module doesn't carry it), so walk the import tree and
    return the first source that actually contains a ``__global__`` entry.
    """
    m = getattr(mod, "mod", mod)
    for cand in _iter_modules(m):
        for fmt in ("cuda", ""):
            try:
                src = cand.inspect_source(fmt)
            except Exception:
                continue
            if isinstance(src, str) and "__global__" in src:
                return src
    raise ValueError(
        "no CUDA source found in module — is it a tir_pipeline='tirx', target='cuda' build?"
    )


def extract_symbols(cuda_src: str) -> list[str]:
    """Return the ``__global__`` kernel symbol names in generated CUDA source.

    The actual entry symbol comes from the PrimFunc's ``global_symbol`` attribute
    (e.g. ``_kernel_kernel``), NOT the IRModule key — so read it back from the
    emitted ``__global__ void [__launch_bounds__(...)] <name>(`` signature rather
    than guessing ``<key>_kernel``.
    """
    seen, out = set(), []
    for m in re.finditer(
        r"__global__\s+void\s+(?:__launch_bounds__\([^)]*\)\s+)?(\w+)\s*\(", cuda_src
    ):
        if m.group(1) not in seen:
            seen.add(m.group(1))
            out.append(m.group(1))
    return out


# ============================================================================
# nvcc / ptxas / cuobjdump pipeline
# ============================================================================


def _run(cmd: list[str], quiet: bool) -> subprocess.CompletedProcess:
    if not quiet:
        print(f"  $ {' '.join(cmd)}")
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    except FileNotFoundError as e:
        # Toolkit executable missing — surface as a failed run so callers collect
        # it into DumpResult.errors instead of crashing with a traceback.
        return subprocess.CompletedProcess(cmd, 127, "", f"executable not found: {e}")


def cuda_to_ptx(cu_path: str, ptx_path: str, arch: str, quiet: bool) -> str:
    """CUDA -> PTX. Returns ``""`` on success, an error message on failure."""
    cmd = [
        _cuda_tool("nvcc"),
        "-ptx",
        f"-arch={arch}",
        "-std=c++17",
        f"-I{_cuda_include()}",
        cu_path,
        "-o",
        ptx_path,
    ]
    r = _run(cmd, quiet)
    if r.returncode != 0:
        return f"nvcc -ptx failed (arch={arch}):\n{r.stderr.strip()}"
    return ""


def cuda_to_cubin(
    cu_path: str, cubin_path: str, arch: str, lineinfo: bool, quiet: bool
) -> tuple[str | None, str]:
    """CUDA -> cubin with ``-Xptxas=-v``. Returns ``(ptxas_log, error)``.

    On success ``(log, "")``; on failure ``(None, error_message)``.
    """
    cmd = [
        _cuda_tool("nvcc"),
        "-cubin",
        f"-arch={arch}",
        "-std=c++17",
        "-Xptxas=-v",
        f"-I{_cuda_include()}",
    ]
    if lineinfo:
        cmd.append("-lineinfo")
    cmd += [cu_path, "-o", cubin_path]
    r = _run(cmd, quiet)
    if r.returncode != 0:
        return None, f"nvcc -cubin failed (arch={arch}):\n{r.stderr.strip()}"
    return r.stderr, ""


# A disassembled SASS instruction is prefixed with its byte offset, e.g. ``/*0a20*/``.
# Absence of any such line means cuobjdump emitted only the cubin header — the cubin
# is empty or (with --function) the requested symbol was not found. cuobjdump exits 0
# and prints the "function not found" note to stderr, so we must detect this ourselves.
_SASS_INSN_RE = re.compile(r"/\*[0-9a-f]+\*/")


def cubin_to_sass(cubin_path: str, sass_path: str, function: str | None, quiet: bool) -> str:
    """cubin -> SASS text file. Returns ``""`` on success, an error message on failure."""
    cmd = [_cuda_tool("cuobjdump"), "--dump-sass"]
    if function:
        cmd += ["--function", function]
    cmd.append(cubin_path)
    r = _run(cmd, quiet)
    if r.returncode != 0:
        return f"cuobjdump --dump-sass failed:\n{r.stderr.strip()}"
    if not _SASS_INSN_RE.search(r.stdout):
        scope = f" for --function {function!r}" if function else ""
        note = r.stderr.strip() or "cubin contains no disassembled instructions"
        return f"cuobjdump produced no SASS{scope}: {note}"
    with open(sass_path, "w") as f:
        f.write(r.stdout)
    return ""


def parse_ptxas(log: str) -> dict:
    """Extract registers / spill / smem / barriers from a ``ptxas -v`` log.

    Caveats: only the **first** entry function's report is parsed (single-kernel
    modules are the common case; ``DumpResult.symbols`` lists all kernels). The
    ``smem`` key is present only for **static** shared memory — kernels using
    ``extern __shared__`` dynamic SMEM (most TIRx GEMM/attention kernels) don't
    report it, so its absence does not mean "0 bytes SMEM".
    """
    # Optional fields may be absent for the first kernel. Searching the entire
    # log would then borrow a resource count from a later, unrelated function.
    entries = list(re.finditer(r"(?m)^ptxas info\s*:\s*Compiling entry function", log))
    if not entries:
        entries = list(re.finditer(r"(?m)^ptxas info\s*:\s*Function properties for", log))
    if entries:
        end = entries[1].start() if len(entries) > 1 else len(log)
        log = log[entries[0].start():end]
    info = {}
    patterns = {
        "registers": r"Used (\d+) registers",
        "spill_stores": r"(\d+) bytes spill stores",
        "spill_loads": r"(\d+) bytes spill loads",
        "stack": r"(\d+) bytes stack frame",
        "smem": r"(\d+) bytes smem",
        "barriers": r"(\d+) barriers",
    }
    for k, pat in patterns.items():
        m = re.search(pat, log)
        if m:
            info[k] = int(m.group(1))
    return info


# ============================================================================
# Programmatic API: compiled module -> CUDA / PTX / SASS
# ============================================================================


@dataclass
class DumpResult:
    """Result of :func:`dump_module` — text of each requested artifact + metadata.

    ``cuda`` is always populated; ``ptx``/``sass`` only when requested **and** the
    nvcc/cuobjdump step succeeded — if a requested stage fails, its field stays
    ``None`` and a message is appended to ``errors`` (check ``d.ok`` /
    ``d.errors`` to tell "not requested" from "requested but failed"). ``ptxas``
    holds the parsed register/spill/smem report and ``ptxas_log`` its raw text.
    ``paths`` maps artifact -> written file, populated only for files actually
    written (requires ``outdir``).
    """

    cuda: str
    arch: str
    symbols: list[str] = field(default_factory=list)
    ptx: str | None = None
    sass: str | None = None
    ptxas: dict = field(default_factory=dict)
    ptxas_log: str | None = None
    paths: dict[str, str] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """True if no requested artifact stage failed."""
        return not self.errors


def dump_module(
    mod,
    *,
    arch: str | None = None,
    ptx: bool = False,
    sass: bool = True,
    lineinfo: bool = True,
    function: str | None = None,
    outdir: str | None = None,
    name: str = "kernel",
    quiet: bool = True,
) -> DumpResult:
    """Dump CUDA / PTX / SASS from a compiled module (the ``tvm.compile`` output).

    ``mod`` is the ``Executable`` returned by ``tvm.compile`` (or a runtime
    ``Module``). CUDA is always returned; ``ptx``/``sass`` toggle those stages,
    which shell out to nvcc/ptxas/cuobjdump. ``arch`` defaults to the local GPU
    (B200 -> ``sm_100a``). With ``outdir`` set, artifacts are written there as
    ``<name>.{cu,ptx,cubin,sass,ptxas.log}`` and their paths returned in
    ``.paths``; otherwise a temp dir is used and removed before returning.

    If a *requested* stage fails (bad arch, unknown ``function``, etc.) its field
    stays ``None`` and a message is appended to ``.errors`` (``.ok`` is then
    ``False``); the failure is printed to stderr only when ``quiet=False``.

    Example::

        ex = tvm.compile(tvm.IRModule({"main": pf}), target="cuda", tir_pipeline="tirx")
        d = dump_module(ex, ptx=True)
        assert d.ok, d.errors
        print(d.sass)                 # SASS text
        print(d.ptxas["registers"])   # 153
    """
    cuda = dump_cuda(mod)
    symbols = extract_symbols(cuda)
    arch = arch or detect_arch()
    res = DumpResult(cuda=cuda, arch=arch, symbols=symbols)

    keep = outdir is not None
    if not (ptx or sass):
        if keep:
            work = Path(outdir)
            work.mkdir(parents=True, exist_ok=True)
            cu = str(work / f"{name}.cu")
            Path(cu).write_text(cuda)
            res.paths["cuda"] = cu
        return res

    work = Path(outdir) if keep else Path(tempfile.mkdtemp(prefix="dump_kernel_"))
    work.mkdir(parents=True, exist_ok=True)

    def p(ext: str) -> str:
        return str(work / f"{name}.{ext}")

    try:
        Path(p("cu")).write_text(cuda)
        if keep:
            res.paths["cuda"] = p("cu")

        if ptx:
            err = cuda_to_ptx(p("cu"), p("ptx"), arch, quiet)
            if err:
                res.errors.append(err)
            else:
                res.ptx = Path(p("ptx")).read_text()
                if keep:
                    res.paths["ptx"] = p("ptx")

        if sass:
            log, err = cuda_to_cubin(p("cu"), p("cubin"), arch, lineinfo, quiet)
            if err:
                res.errors.append(err)
            if log is not None:  # cubin was written
                res.ptxas_log = log
                res.ptxas = parse_ptxas(log)
                if keep:
                    res.paths["cubin"] = p("cubin")
                    Path(p("ptxas.log")).write_text(log)
                    res.paths["ptxas_log"] = p("ptxas.log")
                serr = cubin_to_sass(p("cubin"), p("sass"), function, quiet)
                if serr:
                    res.errors.append(serr)
                else:
                    res.sass = Path(p("sass")).read_text()
                    if keep:
                        res.paths["sass"] = p("sass")
    finally:
        if not keep:
            shutil.rmtree(work, ignore_errors=True)

    if res.errors and not quiet:
        for e in res.errors:
            print(f"[dump_module] {e}", file=sys.stderr)

    return res
