"""Maintenance predicates gate operand reads without hiding lifetime errors."""

import numpy as np
import tvm
from tvm.script import tirx as T

from tirx_harness import numsim, racecheck, synccheck


# TVM's public .pred result ABI is a writable uint32 carrier.
MAINTENANCE_CASES = (
    (True, 1, False),
    (False, 0, False),
    (True, 0, True),
    (False, 1, True),
    (True, None, False),
    (True, None, True),
)


def maintenance_case(preserve=True, layout=1, generic=False, *, invalid=False, count=1):
    """layout=None invalidates/reinitializes instead of querying the layout."""
    pointer = f"barrier.ptr_to([{int(invalid)}])"
    address = (
        f'T.reinterpret("uint64", {pointer})'
        if generic
        else f"T.cuda.cvta_generic_to_shared({pointer})"
    )
    space = "" if generic else ".shared::cta"
    operation = (
        f'T.ptx["mbarrier.check_layout.layout::v{layout}{space}.b64"]('
        f"value[0], address[0], pred=lane == selected, preserve_dst={preserve})"
        if layout is not None
        else f"""T.ptx["mbarrier.inval{space}.b64"](address[0], pred=lane == selected)
    if lane == selected:
        T.ptx["mbarrier.init.layout::v1.shared.b64"](barrier.ptr_to([0]), {count})
    T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.cta_sync()
    if lane == 0:
        T.ptx.mbarrier.arrive.shared.b64(barrier.ptr_to([0]), T.Cast("uint32", T.if_then_else(selected < 0, 2, {count})))
    T.cuda.cta_sync()
    T.ptx.mbarrier.test_wait.parity.shared.b64(value[0], barrier.ptr_to([0]), T.uint32(0))
    if {count} != 1:
        layout_ok = T.alloc_local((1,), "uint32")
        T.ptx["mbarrier.check_layout.layout::v1.shared::cta.b64"](layout_ok[0], barrier.ptr_to([0]))
        value[0] = value[0] * T.Cast("uint32", layout_ok[0] == T.Cast("uint32", selected >= 0))"""
    )
    # Do not observe unspecified inactive GPU outputs for preserve_dst=False.
    output = (
        'T.Cast("uint32", value[0])'
        if preserve or layout is None
        else 'T.if_then_else(lane == selected, T.Cast("uint32", value[0]), T.uint32(0))'
    )
    return tvm.script.from_source(
        f'''@T.prim_func
def kernel(output: T.Buffer((32,), "uint32"), selected: T.int32):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    barrier = T.alloc_shared((1,), "uint64", align=8)
    address = T.alloc_local((1,), "{"uint64" if generic else "uint32"}")
    value = T.alloc_local((1,), "uint32")
    if lane == 0:
        T.ptx.mbarrier.init.shared.b64(barrier.ptr_to([0]), 2)
    T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.cta_sync()
    if lane == selected:
        address[0] = {address}
    value[0] = T.Cast("uint32", T.if_then_else(lane % 2 == 0, -7, 0))
    {operation}
    output[lane] = {output}
''',
        {"T": T},
    )


def maintenance_inputs(selected, preserve=True, layout=1):
    if layout is None:
        expected = np.ones(32, np.uint32)
    else:
        expected = (
            (np.arange(32) % 2 == 0).astype(np.uint32) if preserve else np.zeros(32, np.uint32)
        )
        if selected >= 0:
            expected[selected] = int(layout == 0)
    return {"output": np.zeros(32, np.uint32), "selected": selected}, expected


def test_maintenance_predicates_and_carriers(tmp_path):
    for preserve, layout, generic in MAINTENANCE_CASES:
        # Reinitialization and the v1 maximum belong to the maintenance fixture,
        # not a second kernel that repeats the same invalidate/init/query path.
        kernel = maintenance_case(preserve, layout, generic, count=511 if layout is None else 1)
        module = numsim.transpile(kernel, cache_dir=tmp_path)
        for selected in (-1, 0, 31):
            inputs, expected = maintenance_inputs(selected, preserve, layout)
            for checker in (synccheck, racecheck):
                checker(kernel, inputs).require_clean()
            result = numsim.Engine().run(module, inputs)
            np.testing.assert_array_equal(result.outputs["output"], expected)


def test_maintenance_active_addresses_still_checked():
    for layout in (0, None):
        kernel = maintenance_case(layout=layout, invalid=True)
        for checker in (synccheck, racecheck):
            report = checker(kernel, maintenance_inputs(0, layout=layout)[0])
            assert report.verdict == "error", report.format()
            assert {(f.status, f.kind) for f in report.findings} == {("error", "oob")}
            assert any("raw shared address 8 " in f.message for f in report.findings)
