"""Raw dense `cta_group=2` `tcgen05.mma` joins the TCGEN completion protocol.

The CTA-pair MMA writes both CTAs' TMEM asynchronously, so a reader in either
CTA needs the same `tcgen05.commit` / `mbarrier` handshake the `cta_group=1`
path needs. These cases pin that the CTA-pair variants participate in TCGEN
ordering rather than being treated as an immediate store -- the raw CTA-pair
entries otherwise have numerical coverage only, and a destination written
eagerly would pass every numerical test while losing the ordering edge that
makes a real pipeline safe.

Shared/shared is the case under test; the completion protocol does not depend on
where A comes from, so the TMEM-A form is not duplicated here.
"""

from __future__ import annotations

import numpy as np
import pytest
from tvm.script import tirx as T
from tvm.script.tirx import tile as Tx
from tvm.tirx.layout import ComposeLayout, S, TileLayout, tmem_datapath_layout

from tirx_harness.numsim.checkers import _run_racecheck as racecheck
from tirx_harness.numsim.checkers import _run_synccheck as synccheck


_MMA_F16_32B = ComposeLayout(3, 1, 3, TileLayout(S[(128,)]))


@T.prim_func
def raw_dense_cta2_mma_pipeline(
    left: T.Buffer((128, 16), "float16"),
    right: T.Buffer((32, 16), "float16"),
    output: T.Buffer((128, 32), "float32"),
    with_commit: T.int32,
):
    """One CTA-pair MMA published to both CTAs' readers.

    ``with_commit`` selects the complete protocol: the issuing warp commits the
    CTA-pair work to its own barrier and drains it before the cluster
    rendezvous that releases the readers. With it off the readers reach the
    destination with no completion edge behind them.
    """

    T.device_entry()
    _cluster = T.cluster_id([1])
    cta = T.cta_id_in_cluster([2])
    warp = T.warp_id([2])
    lane = T.lane_id([32])
    barrier = T.alloc_buffer((1,), "uint64", scope="shared")
    left_shared = T.alloc_buffer((64, 16), "float16", scope="shared", layout=_MMA_F16_32B)
    right_shared = T.alloc_buffer((16, 16), "float16", scope="shared", layout=_MMA_F16_32B)
    accumulator = T.decl_buffer(
        (64, 32),
        "float32",
        scope="tmem",
        layout=tmem_datapath_layout("B", 64, 32),
        allocated_addr=0,
    )
    desc_i: T.uint32
    desc_a: T.uint64
    desc_b: T.uint64

    if (warp == 0) and (lane == 0):
        Tx.copy(left_shared[:, :], left[cta * 64 : (cta + 1) * 64, :])
        Tx.copy(right_shared[:, :], right[cta * 16 : (cta + 1) * 16, :])
        T.ptx.mbarrier.init.shared.b64(T.address_of(barrier[0]), 1)
    T.ptx.fence.proxy.async_.shared__cta()
    T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.cluster_sync()

    if (cta == 0) and (warp == 0):
        if lane == 0:
            T.cuda.tcgen05.encode_instr_descriptor(
                T.address_of(desc_i),
                d_dtype="float32",
                a_dtype="float16",
                b_dtype="float16",
                M=128,
                N=32,
                K=16,
                trans_a=False,
                trans_b=False,
                n_cta_groups=2,
            )
            T.cuda.tcgen05.encode_matrix_descriptor(
                T.address_of(desc_a), T.address_of(left_shared[0, 0]), ldo=16, sdo=16, swizzle=1
            )
            T.cuda.tcgen05.encode_matrix_descriptor(
                T.address_of(desc_b), T.address_of(right_shared[0, 0]), ldo=16, sdo=16, swizzle=1
            )
            T.ptx["tcgen05.mma.cta_group::2.kind::f16"](
                T.uint32(0),
                desc_a,
                desc_b,
                desc_i,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                T.ptx.pred(T.uint32(0)),
            )
            if with_commit != 0:
                T.ptx.tcgen05.commit.cta_group__2.mbarrier__arrive__one.shared__cluster.b64(
                    T.address_of(barrier[0])
                )
        if with_commit != 0:
            T.cuda.mbarrier_wait(T.address_of(barrier[0]), 0)
    T.cuda.cluster_sync()

    if (warp == 1) and (lane == 0):
        for row in T.serial(64):
            for col in T.serial(32):
                output[cta * 64 + row, col] = accumulator[row, col]


@pytest.fixture(scope="module")
def native_cache_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("native-racecheck-dense-cta2-mma")


def _inputs(with_commit: bool) -> dict:
    return {
        "left": np.ones((128, 16), dtype=np.float16),
        "right": np.ones((32, 16), dtype=np.float16),
        "output": np.zeros((128, 32), dtype=np.float32),
        "with_commit": np.int32(with_commit),
    }


def _errors(report) -> list:
    """Error-tier findings only."""

    return [finding for finding in report.findings if finding.status == "error"]


def test_committed_dense_cta2_mma_pipeline_is_race_free(native_cache_dir) -> None:
    report = racecheck(
        raw_dense_cta2_mma_pipeline,
        inputs=_inputs(True),
        cache_dir=native_cache_dir,
        max_workers=1,
    )
    report.require_clean()


def test_uncommitted_dense_cta2_mma_pipeline_is_flagged(native_cache_dir) -> None:
    report = racecheck(
        raw_dense_cta2_mma_pipeline,
        inputs=_inputs(False),
        cache_dir=native_cache_dir,
        max_workers=1,
    )
    errors = _errors(report)
    assert errors, report.format()
    # Specifically the CTA-pair MMA store racing the reader, not an incidental
    # protocol failure: the variant's TMEM write must be an async effect the
    # completion handshake publishes.
    assert [finding.details["access_pair"] for finding in errors] == ["write_read"], report.format()
    assert "tmem" in errors[0].message, report.format()


def test_committed_dense_cta2_mma_pipeline_passes_synccheck(tmp_path) -> None:
    synccheck(
        raw_dense_cta2_mma_pipeline,
        inputs=_inputs(True),
        cache_dir=tmp_path,
        max_workers=1,
    ).require_clean()
