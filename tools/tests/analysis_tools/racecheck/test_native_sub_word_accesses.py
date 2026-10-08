"""Checker-mode coverage for the sub-word PTX `ld`/`st` types.

`.b8`/`.u8`/`.s8` move one byte and `.b16`/`.u16`/`.s16` move two, while their
registers are 32 or 16 bits wide. The checkers must see the *memory* width:
a carrier picked from the result dtype would report a four-byte footprint for
`.b8` and manufacture overlaps between neighbouring lanes.
"""

from __future__ import annotations

import numpy as np

from tirx_harness.numsim.checkers import _run_racecheck as racecheck
from tirx_harness.numsim.checkers import _run_synccheck as synccheck
from tvm.script import tirx as T


@T.prim_func
def barrier_ordered_sub_word_store_and_load(output: T.Buffer((64,), "uint32")):
    T.device_entry()
    warp = T.warp_id([2])
    lane = T.lane_id([32])
    bytes8 = T.alloc_buffer((64,), "uint8", scope="shared")
    loaded = T.local_scalar("uint8")
    if warp == 0:
        T.ptx.st.shared.b8(bytes8.ptr_to([lane]), T.cast(lane, "uint8"))
    T.cuda.cta_sync()
    if warp == 1:
        T.ptx.ld.shared.b8(loaded, bytes8.ptr_to([lane]))
        output[lane] = T.cast(loaded, "uint32")


@T.prim_func
def unordered_sub_word_store_and_load(output: T.Buffer((64,), "uint32")):
    T.device_entry()
    warp = T.warp_id([2])
    lane = T.lane_id([32])
    bytes8 = T.alloc_buffer((64,), "uint8", scope="shared")
    loaded = T.local_scalar("uint8")
    if warp == 0:
        T.ptx.st.shared.b8(bytes8.ptr_to([lane]), T.cast(lane, "uint8"))
    if warp == 1:
        T.ptx.ld.shared.b8(loaded, bytes8.ptr_to([lane]))
        output[lane] = T.cast(loaded, "uint32")


def test_racecheck_sees_a_sub_word_access_as_a_one_byte_footprint(tmp_path):
    """`.b8` moves one byte, so its shadow span must be one byte wide.

    A carrier chosen from the 32-bit result dtype instead of the PTX type
    would report a four-byte footprint and manufacture overlaps between
    neighbouring lanes.
    """

    clean = racecheck(
        barrier_ordered_sub_word_store_and_load,
        inputs={"output": np.zeros(64, dtype=np.uint32)},
        cache_dir=tmp_path,
    )
    clean.require_clean()

    report = racecheck(
        unordered_sub_word_store_and_load,
        inputs={"output": np.zeros(64, dtype=np.uint32)},
        cache_dir=tmp_path,
    )
    assert report.verdict == "error"
    spans = {
        finding[side]["span"]["byte_len"]
        for finding in report.to_dict()["native"]["findings"]
        for side in ("prior", "current")
    }
    assert spans == {1}


def test_synccheck_accepts_the_ordered_sub_word_kernel(tmp_path):
    report = synccheck(
        barrier_ordered_sub_word_store_and_load,
        inputs={"output": np.zeros(64, dtype=np.uint32)},
        cache_dir=tmp_path,
    )
    report.require_clean()
