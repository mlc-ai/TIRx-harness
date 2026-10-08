"""Instruction-local TMA overrides: data, descriptor isolation, and validity."""

import numpy as np
import pytest
import tvm
from tvm.script import tirx as T

from tests.numsim.support.execution import run_checked
from tirx_harness import numsim, racecheck, synccheck
from tirx_harness.numsim.transpiler.frontend import analyze
from tests.numsim.support.manifest import call_op_names

ROUTES = ("g2cta", "g2cluster", "s2g", "reduce", "prefetch", "evict_last", "priority")
FORMS = ("address", "dim_b8", "dim_b16", "stride_b8", "stride_b16")
OVERRIDE_CASES = tuple((route, form) for route in ROUTES for form in FORMS)


def override_kernel(
    route,
    form,
    *,
    offset=4,
    coordinate=0,
    dimension=4,
    upper=0,
    issue=True,
    read_early=False,
    report=None,
):
    rank = 2 if "stride" in form or form == "address" else 1
    size = 4 if rank == 1 else 8
    load = route.startswith("g2")
    assert report is None or load
    store = route in {"s2g", "reduce"}
    tokens = ["cp", *(["reduce"] if route == "reduce" else []), "async", "bulk"]
    if route in {"prefetch", "evict_last"}:
        tokens.append("prefetch")
    if route == "priority":
        tokens = ["applypriority", "async", "bulk"]
    tokens += ["tensor", f"{rank}d"]
    if load:
        tokens += [
            "shared::cta" if route == "g2cta" else "shared::cluster",
            "global",
            "mbarrier::complete_tx::bytes",
        ]
        if report is not None:
            tokens.append(f"mbarrier::report::{report}")
    elif store:
        tokens += ["global", "shared::cta"]
        if route == "reduce":
            tokens.append("add")
        tokens.append("bulk_group")
    elif route == "priority":
        tokens += ["global", "bulk_group", "L2::evict_normal"]
    else:
        tokens += ["L2", "global"]
        if route == "evict_last":
            tokens.append("L2::evict_last")
    tokens.append("override::global_address")
    if form != "address":
        tokens.append("override::global_dim_stride" if rank == 2 else "override::global_dim")
    args = ["T.address_of(input_map)", f'T.reinterpret("uint64", replacement.ptr_to([{offset}]))']
    if form != "address":
        dtype = "uint8" if form.endswith("b8") else "uint16"
        args += [f'T.cast({dimension}, "{dtype}")']
        if rank == 2:
            args += [f'T.cast(2, "{dtype}")', "T.uint32(4)", f"T.uint16({upper})"]
    args += [f"T.int32({coordinate})", *(["T.int32(0)"] if rank == 2 else [])]
    if load:
        args = ["shared.ptr_to([0])", *args, "barrier.ptr_to([0])"]
    elif store:
        args.append("shared.ptr_to([0])")
    args.append(f"pred=T.bool({issue})")
    spelling = ".".join(tokens)
    action = f'T.ptx["{spelling}"]({", ".join(args)})'
    completion = ""
    if load:
        completion = (
            f"""
        T.ptx.mbarrier.arrive.expect_tx.shared.b64(barrier.ptr_to([0]), {size * 4})
        T.cuda.mbarrier_wait(barrier.ptr_to([0]), 0)
"""
            if issue
            else ""
        )
    elif store or route == "priority":
        completion = """
        T.ptx.cp.async_.bulk.commit_group()
        T.ptx.cp.async_.bulk.wait_group(0)
"""
    if read_early:
        completion = "        T.ptx.cp.async_.bulk.commit_group()"
    if report is not None and issue:
        completion += f"""
        T.ptx.mbarrier.test_wait.parity.phase_type__primary.shared.b64(
            ready[0], reported[0], barrier.ptr_to([0]), T.uint32(0))
        output[{size}] = T.Cast("float32", ready[0])
        output[{size + 1}] = T.Cast("float32", reported[0])
"""
    observed = "replacement[4 + lane]" if read_early else "shared[lane]"
    drain = "    if lane == 0:\n        T.ptx.cp.async_.bulk.wait_group(0)" if read_early else ""
    return tvm.script.from_source(
        f"""
@T.prim_func
def kernel(input_map: T.TensorMap(), replacement: T.Buffer((32772,), "float32"),
           output: T.Buffer(({size + 2 if report is not None else size},), "float32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_buffer(({size},), "float32", scope="shared")
    barrier = T.alloc_buffer((1,), "uint64", scope="shared")
    {'ready = T.alloc_local((1,), "uint32")' if report is not None else ""}
    {'reported = T.alloc_local((1,), "uint32")' if report is not None else ""}
    if lane < {size}:
        shared[lane] = T.cast(lane + 1, "float32")
    if lane == 0:
        T.ptx["mbarrier.init{".layout::v1" if report is not None else ""}.shared.b64"](barrier.ptr_to([0]), 1)
    T.ptx.fence.proxy.async_.shared__cta()
    T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.cta_sync()
    if lane == 0:
        {action}
{completion}
    T.cuda.cta_sync()
    if lane < {size}:
        output[lane] = {observed}
{drain}
""",
        {"T": T},
    )


def override_inputs(form):
    rank = 2 if "stride" in form or form == "address" else 1
    original = np.full(64, -10, np.float32)
    descriptor = numsim.TensorMap(
        base=original,
        global_shape=(8,) * rank,
        global_strides=() if rank == 1 else (32,),
        box_shape=(4,) if rank == 1 else (4, 2),
        element_strides=(1,) * rank,
    ).numpy()
    return {
        "input_map": descriptor,
        "replacement": np.arange(32772, dtype=np.float32),
        "output": np.zeros(4 if rank == 1 else 8, np.float32),
    }, original


def multiissuer_override_case(route, form, issuer=-1):
    load = route == "load"
    spelling = (
        "cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes"
        if load
        else "cp.reduce.async.bulk.tensor.2d.global.shared::cta.add.tile.bulk_group"
        if route == "reduce"
        else "cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group"
    ) + ".override::global_address.override::global_dim_stride"
    dtype = "uint8" if form == "stride_b8" else "uint16"
    operands = [
        "T.address_of(input_map)",
        'T.reinterpret("uint64", replacement.ptr_to([lane * 16]))',
        f'T.cast(4 - lane, "{dtype}")',
        f'T.cast(2, "{dtype}")',
        "T.uint32(1 + lane)",
        "T.uint16(0)",
        "coordinate[0]",
        "coordinate[0]",
    ]
    if load:
        operands = ["shared.ptr_to([lane * 32])", *operands, "barriers.ptr_to([lane])"]
    else:
        operands.append("shared.ptr_to([lane * 32])")
    completion = (
        """
        ready[0] = T.uint32(0)
        while ready[0] == 0:
            T.ptx.mbarrier.try_wait.parity.shared.b64(
                ready[0], barriers.ptr_to([lane]), T.uint32(0), T.uint32(1))
"""
        if load
        else """
        T.ptx.cp.async_.bulk.commit_group()
        T.ptx.cp.async_.bulk.wait_group.read(0)
        T.ptx.cp.async_.bulk.wait_group(0)
"""
    )
    kernel = tvm.script.from_source(
        f"""
@T.prim_func
def kernel(input_map: T.TensorMap(), replacement: T.Buffer((32800,), "float32"),
           issuer: T.int32, output: T.Buffer((2, 8), "float32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((64,), "float32", scope="shared", align=128)
    barriers = T.alloc_buffer((2,), "uint64", scope="shared")
    ready = T.alloc_local((1,), "uint32")
    coordinate = T.alloc_local((1,), "int32")
    if lane < 2:
        for i in T.serial(8):
            shared[lane * 32 + i] = T.float32(lane * 10 + i + 1)
        T.ptx.mbarrier.init.shared.b64(barriers.ptr_to([lane]), 1)
    T.ptx.fence.proxy.async_.shared__cta()
    T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.cta_sync()
    if {load} and lane < 2 and (issuer < 0 or lane == issuer):
        T.ptx.mbarrier.arrive.expect_tx.shared.b64(barriers.ptr_to([lane]), 32)
    if lane < 2 and (issuer < 0 or lane == issuer):
        coordinate[0] = 0
    T.ptx["{spelling}"]({", ".join(operands)}, pred=lane < 2 and (issuer < 0 or lane == issuer))
    if lane < 2 and (issuer < 0 or lane == issuer):
{completion}
    if lane < 2:
        for i in T.serial(8):
            output[lane, i] = shared[lane * 32 + i]
""",
        {"T": T},
    )
    original = np.full((8, 8), -10, np.float32)
    inputs = dict(
        input_map=numsim.TensorMap(
            original,
            global_shape=(8, 8),
            global_strides=(32,),
            box_shape=(4, 2),
            element_strides=(1, 1),
        ).numpy(),
        replacement=np.arange(32800, dtype=np.float32),
        issuer=issuer,
        output=np.zeros((2, 8), np.float32),
    )
    expected = inputs["replacement"].copy()
    shared = np.arange(8, dtype=np.float32)[None, :] + np.array([[1], [11]], np.float32)
    for lane in range(2):
        if issuer >= 0 and lane != issuer:
            continue
        for row in range(2):
            for column in range(4):
                inside = column < 4 - lane
                index = lane * 16 + row * 4 * (1 + lane) + column
                if load:
                    shared[lane, row * 4 + column] = expected[index] if inside else 0
                elif inside:
                    expected[index] = shared[lane, row * 4 + column] + (
                        expected[index] if route == "reduce" else 0
                    )
    return kernel, inputs, expected, shared


@pytest.mark.parametrize("route,form", OVERRIDE_CASES)
def test_tma_override_data_and_descriptor_isolation(route, form, tmp_path):
    kernel = override_kernel(route, form)
    inputs, original = override_inputs(form)
    descriptor_before = inputs["input_map"].copy()
    before = inputs["replacement"].copy()
    for checker in (synccheck, racecheck):
        checker(kernel, {name: value.copy() for name, value in inputs.items()}).require_clean()
    module = numsim.transpile(kernel, cache_dir=tmp_path)
    result = numsim.Engine().run(module, inputs)
    indices = np.arange(4) + 4
    if inputs["output"].size == 8:
        indices = np.concatenate((indices, indices + (16 if "stride" in form else 8)))
    payload = np.arange(1, len(indices) + 1, dtype=np.float32)
    expected = before.copy()
    if route in {"s2g", "reduce"}:
        expected[indices] = payload + (before[indices] if route == "reduce" else 0)
        assert "replacement" in result.outputs
        # The override redirected the store away from the descriptor's own base,
        # so the TensorMap output shows that base untouched.
        np.testing.assert_array_equal(result.outputs["input_map"], np.float32(-10))
    np.testing.assert_array_equal(inputs["replacement"], expected)
    np.testing.assert_array_equal(
        result.outputs["output"], before[indices] if route.startswith("g2") else payload
    )
    np.testing.assert_array_equal(inputs["input_map"], descriptor_before)
    np.testing.assert_array_equal(original, np.full(64, -10, np.float32))
    assert any("override_" in op_name for op_name in call_op_names(module.spec.kernels[0]))


@pytest.mark.parametrize(
    "kwargs,message",
    [
        ({"offset": 1}, "16-byte aligned"),
        ({"offset": 8}, "128 KiB"),
        ({"coordinate": 1}, "zero coordinates"),
        ({"dimension": 256}, "8-bit"),
        ({"upper": 16}, "unused bits"),
    ],
)
def test_tma_override_rejects_invalid_operands(kwargs, message, tmp_path):
    kernel = override_kernel("g2cta", "stride_b16", **kwargs)
    inputs, _ = override_inputs("stride_b16")
    module = numsim.transpile(kernel, cache_dir=tmp_path)
    source_id = next(
        entry.op_id
        for entry in analyze(kernel).kernels[0].source_map
        if entry.kind == "Call"
        and str(getattr(entry.node.op, "name", "")).endswith("override_global_dim_stride_b16")
    )
    for checker in (synccheck, racecheck):
        report = checker(kernel, inputs)
        assert report.verdict == "error"
        assert message in str(report.to_dict())
        assert report.findings[0].details["operation"]["source_op_id"] == source_id
    with pytest.raises(numsim.NumSimExecutionError, match=message):
        numsim.Engine().run(module, inputs)


def test_tma_override_predicate_skips_invalid_replacement(tmp_path):
    inputs, _ = override_inputs("stride_b16")
    kernel = override_kernel("g2cta", "stride_b16", offset=1, issue=False)
    result = run_checked(kernel, inputs, cache_dir=tmp_path)
    np.testing.assert_array_equal(result.outputs["output"], np.arange(1, 9, dtype=np.float32))


def test_tma_override_keeps_async_destination_race():
    inputs, _ = override_inputs("dim_b16")
    kernel = override_kernel("s2g", "dim_b16", read_early=True)
    report = racecheck(kernel, inputs)
    assert report.verdict == "error"
    assert any(finding.details["access_pair"] == "write_read" for finding in report.findings)
