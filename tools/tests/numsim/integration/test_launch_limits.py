"""CUDA launch-resource limits: shared-memory and cluster-portability modes.

``shared_memory_mode`` and ``portable_cluster_size_mode`` only decide which
launches CUDA accepts, so NumSim executes them like any other launch. A launch
CUDA provably rejects on the kernel's architecture is an error located at its
``T.device_entry``:

* SM107 (measured on VR200, CUDA 13.5): opt-in 232448 B, oversized 334848 B;
  a 327 KiB launch with an oversized mode runs, 328 KiB fails.
* SM100/SM103: opt-in 232448 B; oversized shared memory is treated as absent.
"""

import numpy as np
import pytest

import tvm_tirx
from tests.numsim.support.execution import run_checked, with_cuda_arch
from tirx_harness import numsim, racecheck, synccheck
from tirx_harness.numsim import InvalidLaunchError, NumSimExecutionError
from tirx_harness.numsim.transpiler.frontend import analyze
from tvm_tirx.script import tirx as T

OVERSIZED_SM107 = 334848  # 327 KiB
OPTIN = 232448  # 227 KiB
PORTABLE = 49152


def _source(smem, launch_extra="", *, static_words=0, grid=1, cluster=None):
    launch = f"grid={grid}, block=32, dynamic_smem_bytes={smem}"
    if cluster is not None:
        launch += f", cluster={cluster}"
    static = (
        f"""    static_words = T.alloc_shared(({static_words},), "int32")
    static_words[lane] = lane
"""
        if static_words
        else ""
    )
    return f"""
@T.function
def kernel(output: T.Tensor(({grid * 32},), "int32")):
    T.device_entry(launch=T.cuda.LaunchConfig({launch}{launch_extra}))
    pool = T.alloc_tensor(ty_args=[T.Tensor((0,), "uint8", scope="shared.dyn")])
    words = T.decl_tensor(
        T.ptr_byte_offset(pool.data, 0, ty=T.Ptr("int32", "shared.dyn")),
        ty_args=[T.Tensor((32,), "int32", scope="shared.dyn")],
    )
    lane = T.cuda.lane_id()
    cta = T.cuda.block_idx("x")
{static}    words[lane] = lane + cta
    T.cuda.cta_sync()
    output[cta * 32 + lane] = words[lane]
"""


def _kernel(arch, smem, launch_extra="", **kwargs):
    source = _source(smem, launch_extra, **kwargs)
    kernel = tvm_tirx.script.from_source(source, {"T": T})
    return kernel if arch is None else with_cuda_arch(kernel, arch)


def _launch_line(smem, launch_extra="", **kwargs):
    lines = _source(smem, launch_extra, **kwargs).splitlines()
    return next(index for index, line in enumerate(lines, 1) if "T.device_entry" in line)


@pytest.mark.parametrize(
    "launch_extra",
    [
        *(
            f', shared_memory_mode="{mode}"'
            for mode in (
                "default",
                "require_portable",
                "allow_non_portable",
                "allow_oversized",
                "prefer_oversized",
            )
        ),
        *(
            f', portable_cluster_size_mode="{mode}"'
            for mode in (
                "default",
                "require_portable",
                "allow_non_portable",
            )
        ),
    ],
)
def test_launch_resource_modes_do_not_change_execution(launch_extra):
    spec = analyze(_kernel("sm_107a", 1024, launch_extra))

    assert spec.unsupported == ()


@pytest.mark.parametrize(
    ("arch", "smem", "launch_extra", "kwargs", "reason"),
    [
        # 328 KiB: past the SM107 ceiling in every mode.
        ("sm_107a", 335872, ', shared_memory_mode="allow_oversized"', {}, "maximum oversized"),
        ("sm_107f", 335872, "", {}, "maximum oversized"),
        ("sm_107a", 335872, "", {"grid": 1, "cluster": 1}, "maximum oversized"),
        # Static shared memory counts against the same per-block budget.
        ("sm_107a", OVERSIZED_SM107 - 64, "", {"static_words": 32}, "128 static"),
        # Explicit non-oversized modes cannot reach beyond the opt-in limit.
        ("sm_107a", 300000, ', shared_memory_mode="default"', {}, "opt-in limit"),
        ("sm_107a", 300000, ', shared_memory_mode="allow_non_portable"', {}, "opt-in limit"),
        ("sm_107a", 64 * 1024, ', shared_memory_mode="require_portable"', {}, "portable"),
        ("sm_100a", 64 * 1024, ', shared_memory_mode="require_portable"', {}, "portable"),
        # SM100/SM103 have no oversized configuration.
        ("sm_100a", 300 * 1024, "", {}, "SM100 maximum opt-in"),
        ("sm_103f", OPTIN + 1, ', shared_memory_mode="allow_oversized"', {}, "SM103"),
        ("sm_100a", OPTIN - 64, "", {"static_words": 32}, "SM100 maximum opt-in"),
        # A portable cluster holds at most eight CTAs.
        (
            "sm_107a",
            1024,
            ', portable_cluster_size_mode="require_portable"',
            {"grid": 16, "cluster": 16},
            "portable cluster size 8",
        ),
    ],
)
def test_launch_cuda_rejects_is_an_error_at_the_device_entry(
    arch, smem, launch_extra, kwargs, reason
):
    with pytest.raises(InvalidLaunchError, match=reason) as raised:
        analyze(_kernel(arch, smem, launch_extra, **kwargs))

    assert raised.value.source_span.line == _launch_line(smem, launch_extra, **kwargs)


@pytest.mark.parametrize(
    ("arch", "smem", "launch_extra", "kwargs"),
    [
        ("sm_107a", OVERSIZED_SM107, ', shared_memory_mode="allow_oversized"', {}),
        ("sm_107a", OVERSIZED_SM107, ', shared_memory_mode="prefer_oversized"', {}),
        # Without a mode, TIRx selects an oversized mode for such a launch.
        ("sm_107a", OVERSIZED_SM107, "", {}),
        ("sm_107a", OVERSIZED_SM107, "", {"grid": 1, "cluster": 1}),
        ("sm_107a", OPTIN, ', shared_memory_mode="allow_non_portable"', {}),
        ("sm_107a", PORTABLE, ', shared_memory_mode="require_portable"', {}),
        # Within the opt-in limit, an explicit default mode depends on a
        # function attribute the launch does not determine.
        ("sm_107a", 64 * 1024, ', shared_memory_mode="default"', {}),
        ("sm_100a", OPTIN, "", {}),
        # No limits are known for SM90 or for a kernel without an architecture.
        ("sm_90a", 500000, "", {}),
        (None, 500000, "", {}),
        (
            "sm_107a",
            1024,
            ', portable_cluster_size_mode="allow_non_portable"',
            {"grid": 16, "cluster": 16},
        ),
        ("sm_107a", 1024, ', portable_cluster_size_mode="default"', {"grid": 16, "cluster": 16}),
        (
            "sm_107a",
            1024,
            ', portable_cluster_size_mode="require_portable"',
            {"grid": 8, "cluster": 8},
        ),
    ],
)
def test_launch_within_proven_limits_is_accepted(arch, smem, launch_extra, kwargs):
    spec = analyze(_kernel(arch, smem, launch_extra, **kwargs))

    assert spec.unsupported == ()


def test_runtime_shared_memory_mode_fails_closed():
    kernel = tvm_tirx.script.from_source(
        """
@T.function
def kernel(mode: T.int32, output: T.Tensor((32,), "int32")):
    T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=32, shared_memory_mode=mode))
    output[T.cuda.lane_id()] = 1
""",
        {"T": T},
    )

    spec = analyze(kernel)

    assert any("shared_memory_mode must be a constant" in item for item in spec.unsupported)


def test_rejected_launch_is_an_error_in_every_checker():
    launch_extra = ', shared_memory_mode="allow_oversized"'
    kernel = _kernel("sm_107a", 335872, launch_extra)
    inputs = {"output": np.zeros(32, np.int32)}
    line = _launch_line(335872, launch_extra)

    for checker in (synccheck, racecheck):
        report = checker(kernel, inputs)
        assert report.verdict == "error", report.format()
        (finding,) = report.findings
        assert finding.kind == "invalid_launch"
        assert "335872 bytes" in finding.message
        assert finding.details["source_anchor"]["source_span"]["line"] == line
    with pytest.raises(InvalidLaunchError, match="maximum oversized") as raised:
        numsim.transpile(kernel)
    assert isinstance(raised.value, NumSimExecutionError)


def test_oversized_launch_at_the_sm107_ceiling_is_clean():
    kernel = _kernel("sm_107a", OVERSIZED_SM107, ', shared_memory_mode="allow_oversized"')

    result = run_checked(kernel, {"output": np.zeros(32, np.int32)}, outputs=("output",))

    np.testing.assert_array_equal(result.outputs["output"], np.arange(32, dtype=np.int32))
