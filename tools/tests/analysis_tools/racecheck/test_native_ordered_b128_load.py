"""Observable release/acquire contract for composed ordered b128 loads."""

from __future__ import annotations

import numpy as np

from tirx_harness import numsim
from tvm.script import tirx as T


@T.prim_func
def global_release_to_ordered_b128_acquire(
    data: T.Buffer((1,), "uint64"),
    response: T.Buffer((2,), "uint64"),
    ready: T.Buffer((1,), "uint32"),
    output: T.Buffer((2,), "uint64"),
):
    T.device_entry()
    cta = T.cta_id([2])
    lane = T.lane_id([32])
    loaded = T.alloc_local((1,), "uint128")
    ready_seen = T.local_scalar("uint32")
    published = T.local_scalar("uint64")

    if (cta == 0) and (lane == 0):
        data[0] = T.uint64(0x0123456789ABCDEF)
        response[1] = T.uint64(0x7EDCBA9876543210)
        T.ptx.st.release.gpu.global_.u64(response.ptr_to([0]), T.uint64(1))
        # The relaxed flag only delays the consumer until publication has
        # happened; it supplies no happens-before edge for `data` or response.
        T.ptx.st.relaxed.gpu.global_.u32(ready.ptr_to([0]), T.uint32(1))
    elif (cta == 1) and (lane == 0):
        ready_seen = T.uint32(0)
        T.cuda.wait_until(
            ready_seen, ready.ptr_to([0]), ready_seen != T.uint32(0), "gpu", "global",
        )
        # `ready` is published with a relaxed store *after* the release store,
        # and release orders only what precedes it, so seeing `ready` does not
        # mean the response landed. What the consumer is entitled to is the
        # write its predicate accepts, so it says so: the acquiring wait on the
        # response's low half is the handoff, and the `.b128` read that follows
        # takes the pair the wait already made visible.
        published = T.uint64(0)
        T.cuda.wait_until(
            published, response.ptr_to([0]), published != T.uint64(0), "gpu", "global",
        )
        T.ptx.ld.relaxed.gpu.global_.b128(loaded[0], response.ptr_to([0]))
        output[0] = data[0]
        output[1] = loaded.view("uint64")[1]


def test_ordered_b128_acquire_preserves_release_handoff(tmp_path):
    module = numsim.transpile(
        global_release_to_ordered_b128_acquire,
        cache_dir=tmp_path,
        _analysis_capable=True,
        _analysis_checker="racecheck",
    )
    result = numsim.Engine(max_workers=2).run_racecheck_phase(
        module,
        {
            "data": np.zeros(1, dtype=np.uint64),
            "response": np.zeros(2, dtype=np.uint64),
            "ready": np.zeros(1, dtype=np.uint32),
            "output": np.zeros(2, dtype=np.uint64),
        },
    )

    assert result.verdict == "clean"
    assert result.findings == []
    assert result.incomplete == []
