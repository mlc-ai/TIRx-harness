"""Unit tests for generated-source inspection helpers."""

import subprocess
import sys

import pytest

import tirx_harness.dump_kernel as inspection
from tirx_harness.dump_kernel import dump_cuda, extract_symbols, parse_ptxas


class _Module:
    def __init__(self, source: str = "", imports: list["_Module"] | None = None) -> None:
        self._source = source
        self.imports = imports or []

    def inspect_source(self, _format: str) -> str:
        return self._source


class _Executable:
    def __init__(self, module: _Module) -> None:
        self.mod = module


def test_dump_cuda_walks_imported_modules() -> None:
    cuda = 'extern "C" __global__ void _kernel_kernel() {}'
    executable = _Executable(_Module(imports=[_Module(cuda)]))

    assert dump_cuda(executable) == cuda


def test_extract_symbols_handles_launch_bounds_and_duplicates() -> None:
    cuda = """
    __global__ void __launch_bounds__(128) first_kernel() {}
    __global__ void second_kernel() {}
    __global__ void first_kernel() {}
    """

    assert extract_symbols(cuda) == ["first_kernel", "second_kernel"]


def test_parse_ptxas_extracts_resource_counts() -> None:
    log = """
    ptxas info    : Used 153 registers, 2 barriers, 16 bytes smem
    ptxas info    : 32 bytes stack frame, 8 bytes spill stores, 4 bytes spill loads
    """

    assert parse_ptxas(log) == {
        "registers": 153,
        "spill_stores": 8,
        "spill_loads": 4,
        "stack": 32,
        "smem": 16,
        "barriers": 2,
    }


def test_parse_ptxas_does_not_borrow_optional_fields_from_later_kernels() -> None:
    # CUDA 13.0 SM103a logs from real plain/shared kernels: the first report has
    # no static SMEM, while the second has 128 bytes. Preserve that distinction.
    first = """ptxas info    : Compiling entry function 'plain' for 'sm_103a'
ptxas info    : Function properties for plain
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 10 registers, used 0 barriers
ptxas info    : Compile time = 2.481 ms
"""
    second = """ptxas info    : Compiling entry function 'shared' for 'sm_103a'
ptxas info    : Function properties for shared
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 12 registers, used 1 barriers, 128 bytes smem
"""
    assert parse_ptxas(first + second) == parse_ptxas(first)
    assert "smem" not in parse_ptxas(first + second)
    assert parse_ptxas(second + first) == parse_ptxas(second)


def test_parse_ptxas_scopes_reports_without_compile_headers() -> None:
    log = """ptxas info    : Function properties for plain
ptxas info    : Used 10 registers
ptxas info    : Function properties for later
ptxas info    : 32 bytes stack frame, 8 bytes spill stores, 4 bytes spill loads
ptxas info    : Used 12 registers, 1 barriers, 128 bytes smem
"""
    assert parse_ptxas(log) == {"registers": 10}


@pytest.mark.parametrize("stage", ["ptx", "cubin", "sass"])
def test_dump_module_reports_tool_timeout(monkeypatch, stage) -> None:
    # Use a real child process, but shorten the timeout and replace the CUDA
    # tool so this failure-path regression needs neither a GPU nor a toolkit.
    run = subprocess.run

    def stalled_tool(cmd, **kwargs):
        kwargs["timeout"] = 0.5
        return run(
            [
                sys.executable,
                "-c",
                "import sys,time; print('partial diagnostic', file=sys.stderr, flush=True); "
                "time.sleep(60)",
            ],
            **kwargs,
        )

    monkeypatch.setattr(inspection.subprocess, "run", stalled_tool)
    if stage == "sass":
        # Reach the disassembler independently of the preceding compiler.
        monkeypatch.setattr(inspection, "cuda_to_cubin", lambda *args: ("", ""))
    module = _Module('extern "C" __global__ void test_kernel() {}')
    result = inspection.dump_module(module, arch="sm_100a", ptx=stage == "ptx", sass=stage != "ptx")

    assert not result.ok
    assert result.cuda == module._source
    assert result.ptx is None
    assert result.sass is None
    assert len(result.errors) == 1
    assert "timed out" in result.errors[0]
    assert "partial diagnostic" in result.errors[0]


@pytest.mark.parametrize("output", [b"partial", "partial", None])
def test_tool_timeout_normalizes_partial_output(monkeypatch, output) -> None:
    def timeout(cmd, **kwargs):
        raise subprocess.TimeoutExpired(cmd, kwargs["timeout"], output=output, stderr=output)

    monkeypatch.setattr(inspection.subprocess, "run", timeout)
    result = inspection._run(["nvcc", "-ptx"], quiet=True)

    assert result.returncode == 124
    assert result.stdout == ("partial" if output is not None else "")
    assert isinstance(result.stderr, str)
    assert "timed out after 300 seconds: nvcc -ptx" in result.stderr
    if output is not None:
        assert result.stderr.endswith("partial")
