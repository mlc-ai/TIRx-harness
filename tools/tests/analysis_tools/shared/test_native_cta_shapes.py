"""Synccheck and Racecheck on multi-dimensional and partial-warp CTAs.

A CTA's threads are linear ids `x + y * bx + z * bx * by` in warps of 32, so
the shapes below put rows across warp boundaries and leave a partial last warp.
"""

from __future__ import annotations

import ml_dtypes
import numpy as np
import pytest

import tvm_tirx
from tirx_harness import numsim
from tirx_harness.numsim.checkers import _run_racecheck as racecheck
from tirx_harness.numsim.checkers import _run_synccheck as synccheck
from tvm_tirx.script import tirx as T


def _verdicts(kernel, inputs, cache_dir, checks=(synccheck, racecheck)):
    reports = [
        check(kernel, inputs=dict(inputs), cache_dir=cache_dir, max_workers=4) for check in checks
    ]
    return [report.verdict for report in reports], "\n".join(r.format() for r in reports)


@pytest.fixture(scope="module")
def cache_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("cta-shapes")


# The two RL kernels that were checker_unsupported (numsim-coverage-fixtures
# episodes 3280666.1530 group-0-13 and 3280666.1311 group-2-53), regenerated
# from their source text: C = A @ B.T over 128x128 BF16 with 128 unrolled
# FP32 products, by a 16x16-thread CTA per tile or by one thread.
TENSOR = 'T.Tensor((128, 128), "bfloat16")'
BLOCK_GEMM = """
@T.function
def gemm(A: {tensor}, B: {tensor}, C: {tensor}):
    with T.device_entry(launch=T.cuda.LaunchConfig(grid=(8, 8, 1), block=(16, 16, 1))):
        m = T.cuda.block_idx("y") * 16 + T.cuda.thread_idx("y")
        n = T.cuda.block_idx("x") * 16 + T.cuda.thread_idx("x")
        acc = T.float32(0.0)
{body}
        C[m, n] = T.cast(acc, ty="bfloat16")
"""
SERIAL_GEMM = """
@T.function
def gemm(A: {tensor}, B: {tensor}, C: {tensor}):
    T.device_entry(launch=T.cuda.LaunchConfig(grid=(1, 1), block=(1, 1)))
    for i in T.serial(128):
        for j in T.serial(128):
{body}
            C[i, j] = T.cast({total}, "bfloat16")
"""


def _rl_gemm(serial: bool):
    if serial:
        body = "\n".join(
            f'            p{k}_ = T.cast(A[i, {k}], "float32") * T.cast(B[j, {k}], "float32")'
            for k in range(128)
        )
        total = " + ".join(f"p{k}_" for k in range(128))
        source = SERIAL_GEMM.format(tensor=TENSOR, body=body, total=total)
    else:
        body = "\n".join(
            f'        acc = acc + T.cast(A[m, {k}], ty="float32") * T.cast(B[n, {k}], ty="float32")'
            for k in range(128)
        )
        source = BLOCK_GEMM.format(tensor=TENSOR, body=body)
    return tvm_tirx.script.from_source(source, {"T": T})


@pytest.mark.parametrize("serial", [False, True])
def test_rl_kernels_are_clean_and_exact(cache_dir, serial):
    # Values in {-1, 0, 1} keep every sum exact in FP32 and BF16.
    kernel = _rl_gemm(serial)
    rng = np.random.default_rng(0)
    a, b = (rng.integers(-1, 2, (128, 128)).astype(ml_dtypes.bfloat16) for _ in range(2))
    inputs = {"A": a, "B": b, "C": np.zeros((128, 128), ml_dtypes.bfloat16)}
    compiled = numsim.transpile(kernel, cache_dir=cache_dir)
    result = numsim.Engine().run(compiled, dict(inputs), outputs=("C",)).outputs["C"]
    reference = a.astype(np.float32) @ b.astype(np.float32).T
    np.testing.assert_array_equal(result.astype(np.float32), reference)
    verdicts, text = _verdicts(kernel, inputs, cache_dir)
    assert verdicts == ["clean", "clean"], text


def _row_kernel(block, body):
    bx, by, bz = block
    threads = bx * by * bz
    return tvm_tirx.script.from_source(
        f"""
@T.function
def rows(output: T.Tensor(({threads},), "int32"), flag: T.int32):
    T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block={block}))
    x = T.cuda.thread_idx("x")
    y = T.cuda.thread_idx("y")
    linear = x + y * {bx}
    shared = T.alloc_shared(({threads},), "int32")
{body}
""",
        {"T": T},
    )


# Row y reads row y + 1. (24, 4): rows straddle warps; (10, 10): the last warp
# is partial. `flag` selects whether the barrier is executed.
NEIGHBOUR = """    shared[linear] = linear * 3 + 1
    if flag == 1:
        T.cuda.cta_sync()
    output[linear] = shared[(linear + {bx}) % {threads}]"""


@pytest.mark.parametrize("block", [(24, 4, 1), (10, 10, 1)])
def test_neighbour_rows_need_a_barrier(cache_dir, block):
    threads = block[0] * block[1]
    kernel = _row_kernel(block, NEIGHBOUR.format(bx=block[0], threads=threads))
    synced = {"output": np.zeros(threads, np.int32), "flag": np.int32(1)}
    module = numsim.transpile(kernel, cache_dir=cache_dir)
    output = numsim.Engine().run(module, dict(synced), outputs=("output",)).outputs["output"]
    np.testing.assert_array_equal(output, (np.arange(threads) + block[0]) % threads * 3 + 1)
    verdicts, text = _verdicts(kernel, synced, cache_dir)
    assert verdicts == ["clean", "clean"], text
    racy = {**synced, "flag": np.int32(0)}
    verdicts, text = _verdicts(kernel, racy, cache_dir, checks=(racecheck,))
    assert verdicts == ["error"], text


def test_rows_writing_one_shared_word_race(cache_dir):
    # Every row writes shared[x]: threads that differ only in y race,
    # within one warp (rows 0 and 1 share warp 0) and across warps.
    kernel = _row_kernel((24, 4, 1), "    shared[x] = y\n    output[linear] = shared[x]")
    inputs = {"output": np.zeros(96, np.int32), "flag": np.int32(0)}
    verdicts, text = _verdicts(kernel, inputs, cache_dir, checks=(racecheck,))
    assert verdicts == ["error"], text


def test_cta_sync_divergent_on_y_is_reported(cache_dir):
    # Partial warp 1 holds rows 8 and 9 and diverges at the barrier.
    block, condition, threads = (4, 10, 1), "y < 9", 40
    body = f"""    shared[linear] = linear
    if {condition}:
        T.cuda.cta_sync()
    output[linear] = shared[{threads - 1} - linear]"""
    inputs = {"output": np.zeros(threads, np.int32), "flag": np.int32(0)}
    verdicts, text = _verdicts(_row_kernel(block, body), inputs, cache_dir)
    assert "clean" not in verdicts, text


@T.function
def tiled_shared_gemm_2d(
    A: T.Tensor((32, 32), "float32"),
    B: T.Tensor((32, 32), "float32"),
    C: T.Tensor((32, 32), "float32"),
    second_sync: T.int32,
):
    T.device_entry(launch=T.cuda.LaunchConfig(grid=(2, 2), block=(16, 16)))
    tx = T.cuda.thread_idx("x")
    ty = T.cuda.thread_idx("y")
    row = T.cuda.block_idx("y") * 16 + ty
    col = T.cuda.block_idx("x") * 16 + tx
    a_tile = T.alloc_shared((16, 16), "float32")
    b_tile = T.alloc_shared((16, 16), "float32")
    acc = T.local_scalar("float32")
    acc = T.float32(0)
    for tile in T.serial(2):
        a_tile[ty, tx] = A[row, tile * 16 + tx]
        b_tile[ty, tx] = B[tile * 16 + ty, col]
        T.cuda.cta_sync()
        for k in T.serial(16):
            acc = acc + a_tile[ty, k] * b_tile[k, tx]
        if second_sync == 1:
            T.cuda.cta_sync()
    C[row, col] = acc


def test_tiled_shared_gemm_2d(cache_dir):
    rng = np.random.default_rng(3)
    a, b = (rng.integers(-3, 4, (32, 32)).astype(np.float32) for _ in range(2))
    inputs = {"A": a, "B": b, "C": np.zeros((32, 32), np.float32), "second_sync": np.int32(1)}
    module = numsim.transpile(tiled_shared_gemm_2d, cache_dir=cache_dir)
    result = numsim.Engine().run(module, dict(inputs), outputs=("C",))
    np.testing.assert_array_equal(result.outputs["C"], a @ b)
    verdicts, text = _verdicts(tiled_shared_gemm_2d, inputs, cache_dir)
    assert verdicts == ["clean", "clean"], text
    # Without the second barrier the next tile overwrites shared memory being read.
    racy = {**inputs, "second_sync": np.int32(0)}
    verdicts, text = _verdicts(tiled_shared_gemm_2d, racy, cache_dir, checks=(racecheck,))
    assert verdicts == ["error"], text
