"""Regression coverage for one mbarrier pointer reused by TMA copies.

A TIR `let` of an mbarrier pointer produces one reusable Rust `PhysicalPtr`
binding. Every TMA use may clone that value through the existing owned
`Address` ABI; the clone is cheap because the pointer state is shared and
derivation uses copy-on-write. An accidental move would fail the native build
with rustc E0382 before any analysis runs.
"""

from __future__ import annotations

import ml_dtypes
import numpy as np
import tvm

from tirx_harness.numsim.checkers import _run_synccheck as synccheck
from tvm.backend.cuda.tile_primitive.tma_utils import (
    SwizzleMode,
    mma_shared_layout,
)
from tvm.script import tirx as T
from tvm.script.tirx import tile as Tx
from tvm.tirx.lang.pipeline import PipelineState, TMABar

bf16 = tvm.DataType("bfloat16")
LAYOUT = mma_shared_layout(bf16, SwizzleMode.SWIZZLE_128B_ATOM, (64, 128))


@T.prim_func
def multi_copy_shared_tma_mbar(
    source: T.Buffer((3, 64, 128), "bfloat16"),
):
    T.device_entry()
    tx = T.thread_id([128])
    pool = T.SMEMPool()
    ready = TMABar(pool, 1)
    pool.move_base_to(1024)
    first = pool.alloc((64, 128), bf16, layout=LAYOUT)
    second = pool.alloc((64, 128), bf16, layout=LAYOUT)
    third = pool.alloc((64, 128), bf16, layout=LAYOUT)
    pool.commit()
    ready.init(1)

    T.ptx.fence.proxy.async_.shared__cta()
    T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.cta_sync()
    state = PipelineState(1, 0)
    if tx == 0:
        shared_mbar = ready.ptr_to([state.stage])
        Tx.copy_async(
            first[:, :],
            source[0, :, :],
            dispatch="tma_auto",
            mbar=shared_mbar,
        )
        Tx.copy_async(
            second[:, :],
            source[1, :, :],
            dispatch="tma_auto",
            mbar=shared_mbar,
        )
        Tx.copy_async(
            third[:, :],
            source[2, :, :],
            dispatch="tma_auto",
            mbar=shared_mbar,
        )
        ready.arrive(state.stage, 3 * 64 * 128 * 2)
    ready.wait(state.stage, state.phase)


def _assert_clean(report) -> None:
    report.require_clean()
    assert report.verdict == "clean"
    assert report.findings == []
    native = report.to_dict()["native"]
    assert native["verdict"] == "clean"
    assert native["incomplete"] == []


def test_multi_copy_shared_tma_mbar(tmp_path):
    report = synccheck(
        multi_copy_shared_tma_mbar,
        inputs={"source": np.zeros((3, 64, 128), dtype=ml_dtypes.bfloat16)},
        cache_dir=tmp_path,
        max_workers=1,
    )

    _assert_clean(report)
