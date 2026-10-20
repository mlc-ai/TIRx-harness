"""PTX 8.7.1 exempts strong flag accesses without publishing unrelated payloads."""

import numpy as np
import pytest
import tvm
from tvm.script import tirx as T

from tirx_harness.numsim.checkers import _run_racecheck as racecheck


@pytest.mark.parametrize("max_workers", [1, 2])
@pytest.mark.parametrize("order", ["relaxed", "acquire"])
@pytest.mark.parametrize("writer", ["store", "atomic"])
def test_strong_flag_read_and_write_need_no_happens_before(writer, order, max_workers):
    publication = (
        "T.ptx.st.release.gpu.global_.b32(flag.ptr_to([0]), T.uint32(1))"
        if writer == "store" else
        "T.ptx.atom.release.gpu.global_.add.u32(seen, flag.ptr_to([0]), T.uint32(1))"
    )
    kernel = tvm.script.from_source(
        f"""
@T.prim_func
def flags(flag: T.Buffer((1,), "uint32"), output: T.Buffer((1,), "uint32")):
    T.device_entry()
    cta = T.cta_id([2])
    lane = T.lane_id([32])
    seen = T.local_scalar("uint32")
    if lane == 0:
        if cta == 0:
            {publication}
        else:
            T.ptx.ld.{order}.gpu.global_.b32(seen, flag.ptr_to([0]))
            output[0] = seen
""",
        {"T": T},
    )
    inputs = {"flag": np.zeros(1, dtype=np.uint32), "output": np.zeros(1, dtype=np.uint32)}
    racecheck(kernel, inputs=inputs, max_workers=max_workers).require_clean()


@pytest.mark.parametrize("max_workers", [1, 2])
def test_strong_flag_read_does_not_acquire_an_unobserved_payload(max_workers):
    kernel = tvm.script.from_source(
        """
@T.prim_func
def payload(flag: T.Buffer((2,), "uint32"), data: T.Buffer((1,), "uint32"),
            output: T.Buffer((1,), "uint32")):
    T.device_entry()
    cta = T.cta_id([2])
    lane = T.lane_id([32])
    seen = T.local_scalar("uint32")
    if lane == 0:
        if cta == 0:
            data[0] = T.uint32(41)
            T.ptx.st.release.gpu.global_.b32(flag.ptr_to([0]), T.uint32(1))
        else:
            T.ptx.ld.acquire.gpu.global_.b32(seen, flag.ptr_to([1]))
            output[0] = data[0] + seen
""",
        {"T": T},
    )
    inputs = {"flag": np.ones(2, dtype=np.uint32), "data": np.zeros(1, dtype=np.uint32),
              "output": np.zeros(1, dtype=np.uint32)}
    report = racecheck(kernel, inputs=inputs, max_workers=max_workers)

    assert report.verdict == "error", report.format()
    assert any(finding["kind"] == "data_race"
               for finding in report.native_payload["findings"])
