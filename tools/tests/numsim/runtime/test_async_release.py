"""Global async release: payloads, independent completion, and publication."""

import numpy as np
import pytest
from tvm.script import tirx as T

from tests.numsim.support.execution import run_checked
from tirx_harness import racecheck

# Byte widths and carrier classes are independent for a store. Signed arithmetic
# remains covered separately by the 32/64-bit reduction cases below.
STORE_TYPES = ("b8", "u16", "s32", "b64", "f32", "f64")


def release_kernel(ptx_type="u32", *, reduction=False, consume=False):
    dtype = {"b": "uint", "u": "uint", "s": "int", "f": "float"}[ptx_type[0]] + ptx_type[1:]
    instruction = (
        f"{'red' if reduction else 'st'}_async.release.gpu.global."
        f"{'add.' if reduction else ''}{ptx_type}"
    )

    @T.prim_func
    def kernel(data: T.Buffer((32,), dtype), output: T.Buffer((32,), dtype)):
        T.device_entry()
        _warp = T.warp_id([1])
        lane = T.lane_id([32])
        T.ptx[instruction](data.ptr_to([lane]), T.cast(lane - 3, dtype), pred=lane < 16)
        if consume:
            T.ptx.cp.async_.bulk.commit_group()
            T.ptx.cp.async_.bulk.wait_group(0)
            output[lane] = data[lane]

    return kernel, dtype


@pytest.mark.parametrize(
    "reduction,ptx_type",
    [(False, t) for t in STORE_TYPES] + [(True, t) for t in ("u32", "s32", "u64", "s64")],
)
def test_async_release_payloads(reduction, ptx_type, tmp_path):
    kernel, dtype = release_kernel(ptx_type, reduction=reduction)
    args = {"data": np.full(32, 7, dtype), "output": np.zeros(32, dtype)}
    actual = run_checked(kernel, args, cache_dir=tmp_path)
    expected = args["data"].copy()
    expected[:16] = (np.arange(16) + (4 if reduction else -3)).astype(dtype)
    np.testing.assert_array_equal(actual.outputs["data"], expected)


@pytest.mark.parametrize("reduction", [False, True])
def test_bulk_wait_does_not_acquire_async_release(reduction):
    kernel, dtype = release_kernel(consume=True, reduction=reduction)
    report = racecheck(kernel, {"data": np.zeros(32, dtype), "output": np.zeros(32, dtype)})
    assert report.verdict == "error"
    assert any(f.details["access_pair"] in {"write_read", "read_write"} for f in report.findings)


def publication_kernel(*, after=False, shared=False, reduction=False):
    instruction = (
        "red_async.release.gpu.global.add.u32" if reduction else "st_async.release.gpu.global.u32"
    )

    @T.prim_func
    def kernel(
        data: T.Buffer((1,), "uint32"),
        flag: T.Buffer((1,), "uint32"),
        output: T.Buffer((1,), "uint32"),
    ):
        T.device_entry()
        warp = T.warp_id([2])
        lane = T.lane_id([32])
        scratch = T.alloc_shared((1,), "uint32")
        if lane == 0:
            if warp == 0:
                if not after:
                    if shared:
                        scratch[0] = T.uint32(42)
                    else:
                        data[0] = T.uint32(42)
                T.ptx[instruction](flag.ptr_to([0]), T.uint32(1))
                if after:
                    if shared:
                        scratch[0] = T.uint32(42)
                    else:
                        data[0] = T.uint32(42)
            else:
                observed = T.alloc_local((1,), "uint32")
                observed[0] = T.uint32(0)
                T.cuda.wait_until(
                    observed[0],
                    flag.ptr_to([0]),
                    observed[0] != T.uint32(0),
                    scope="gpu",
                    ptx_type="u32",
                )
                if shared:
                    output[0] = scratch[0]
                else:
                    output[0] = data[0]

    return kernel


@pytest.mark.parametrize("shared,reduction", [(False, False), (True, False), (False, True)])
def test_async_release_publishes_only_pre_issue_work(shared, reduction, tmp_path):
    args = {name: np.zeros(1, np.uint32) for name in ("data", "flag", "output")}
    kernel = publication_kernel(shared=shared, reduction=reduction)
    actual = run_checked(kernel, args, cache_dir=tmp_path)
    np.testing.assert_array_equal(actual.outputs["output"], [42])
    report = racecheck(publication_kernel(after=True, shared=shared, reduction=reduction), args)
    assert report.verdict == "error"
    assert any(f.details["access_pair"] in {"read_write", "write_read"} for f in report.findings)
