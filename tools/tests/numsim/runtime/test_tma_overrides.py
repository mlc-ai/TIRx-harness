"""Instruction-local TMA overrides: data, descriptor isolation, and validity."""

import numpy as np
import pytest

import tvm_tirx
from tests.numsim.support.execution import run_checked
from tests.numsim.support.manifest import call_op_names
from tirx_harness import numsim, racecheck, synccheck
from tirx_harness.numsim.transpiler.frontend import analyze
from tvm_tirx.script import tirx as T

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
    args = [
        "T.address_of(input_map)",
        f'T.reinterpret(replacement.ptr_to([{offset}]), ty="uint64")',
    ]
    if form != "address":
        dtype = "uint8" if form.endswith("b8") else "uint16"
        # Override dimensions are encoded as size - 1, like the descriptor's.
        args += [f'T.cast({dimension - 1}, "{dtype}")']
        if rank == 2:
            args += [f'T.cast(1, "{dtype}")', "T.uint32(4)", f"T.uint16({upper})"]
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
    return tvm_tirx.script.from_source(
        f"""
@T.function
def kernel(input_map: T.Ptr(T.TensorMap), replacement: T.Tensor((32772,), "float32"),
           output: T.Tensor(({size + 2 if report is not None else size},), "float32")):
    T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=32))
    lane = T.cuda.lane_id()
    shared = T.alloc_shared(({size},), "float32")
    barrier = T.alloc_shared((1,), "uint64")
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
        'T.reinterpret(replacement.ptr_to([lane * 16]), ty="uint64")',
        f'T.cast(3 - lane, "{dtype}")',
        f'T.cast(1, "{dtype}")',
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
    kernel = tvm_tirx.script.from_source(
        f"""
@T.function
def kernel(input_map: T.Ptr(T.TensorMap), replacement: T.Tensor((32800,), "float32"),
           issuer: T.int32, output: T.Tensor((2, 8), "float32")):
    T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=32))
    lane = T.cuda.lane_id()
    shared = T.alloc_shared((64,), "float32", align=128)
    barriers = T.alloc_shared((2,), "uint64")
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
                # SM107 bounds loads by the overridden dimension per element;
                # stores and reductions round the inner bound up to 16 bytes,
                # which covers this 4-float box row for every lane.
                inside = column < 4 - lane or not load
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


def partial_dimension_expectation(form, dimension):
    """Loaded values for a 4-wide box whose overridden inner dimension is `dimension`."""

    row = np.where(np.arange(4) < dimension, np.arange(4, 8, dtype=np.float32), 0)
    if "stride" not in form:
        return row
    return np.concatenate((row, np.where(np.arange(4) < dimension, row + 16, 0)))


# Recorded on an SM107 GPU (NVIDIA VR NVL72 ES): the dimension operand d loads
# d + 1 elements, so a dimension below the box width zero-fills the remaining columns.
@pytest.mark.parametrize("form", ("dim_b8", "dim_b16", "stride_b8", "stride_b16"))
@pytest.mark.parametrize("dimension", (1, 2, 3, 4))
def test_tma_override_dimension_operand_encodes_size_minus_one(form, dimension, tmp_path):
    kernel = override_kernel("g2cta", form, dimension=dimension)
    inputs, _ = override_inputs(form)
    result = run_checked(kernel, inputs, cache_dir=tmp_path)
    np.testing.assert_array_equal(
        result.outputs["output"], partial_dimension_expectation(form, dimension)
    )


@pytest.mark.parametrize("route", ("s2g", "reduce"))
@pytest.mark.parametrize("form", ("dim_b8", "stride_b16"))
@pytest.mark.parametrize("dimension", (1, 3))
def test_tma_override_store_inner_dimension_rounds_up_to_16_bytes(route, form, dimension, tmp_path):
    # The 4-float (16-byte) box row is one store chunk: any inner bound
    # inside it writes the whole row (SM107 measurement, #47).
    kernel = override_kernel(route, form, dimension=dimension)
    inputs, _ = override_inputs(form)
    before = inputs["replacement"].copy()
    run_checked(kernel, inputs, cache_dir=tmp_path)
    indices = np.arange(4) + 4
    if "stride" in form:
        indices = np.concatenate((indices, indices + 16))
    expected = before.copy()
    payload = np.arange(1, len(indices) + 1, dtype=np.float32)
    expected[indices] = payload + (before[indices] if route == "reduce" else 0)
    np.testing.assert_array_equal(inputs["replacement"], expected)


@pytest.mark.parametrize(
    "kwargs,message",
    [
        ({"offset": 1}, "16-byte aligned"),
        ({"offset": 8}, "128 KiB"),
        ({"coordinate": 1}, "zero coordinates"),
        ({"dimension": 257}, "8-bit"),
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


# A 16 x 8 fp32 box (64-byte rows) moved through a 128-byte row pitch, so a
# write past the box's inner edge would land in the visible row gap.
CLIP_BOX = (16, 8)
CLIP_PITCH = 32
CLIP_SPELLINGS = {
    "load": "cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes",
    "s2g": "cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group",
    "reduce": "cp.reduce.async.bulk.tensor.2d.global.shared::cta.add.tile.bulk_group",
}


def clip_kernel(route, *, override=True):
    """One CTA moves the box at coordinates (0, 0). With ``override`` the
    instruction redirects it to ``y`` with the runtime size-minus-one
    dimension operands ``d0``/``d1``; otherwise it uses the descriptor."""

    spelling = CLIP_SPELLINGS[route]
    operands = ["T.address_of(m)"]
    if override:
        spelling += ".override::global_address.override::global_dim_stride"
        operands += [
            'T.reinterpret(y.ptr_to([0]), ty="uint64")',
            'T.cast(d0, "uint8")',
            'T.cast(d1, "uint8")',
            f"T.uint32({CLIP_PITCH * 4 // 16})",
            "T.uint16(0)",
        ]
    operands += ["T.int32(0)", "T.int32(0)"]
    size = CLIP_BOX[0] * CLIP_BOX[1]
    if route == "load":
        operands = ["tile.ptr_to([0])", *operands, "barrier.ptr_to([0])"]
        completion = f"""
        T.ptx.mbarrier.arrive.expect_tx.shared.b64(barrier.ptr_to([0]), {size * 4})
        T.cuda.mbarrier_wait(barrier.ptr_to([0]), 0)"""
    else:
        operands.append("tile.ptr_to([0])")
        completion = """
        T.ptx.cp.async_.bulk.commit_group()
        T.ptx.cp.async_.bulk.wait_group(0)"""
    return tvm_tirx.script.from_source(
        f"""
@T.function
def kernel(m: T.Ptr(T.TensorMap), y: T.Tensor((32768 + {CLIP_PITCH * 8},), "float32"),
           out: T.Tensor(({size},), "float32"), d0: T.int32, d1: T.int32):
    T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=32))
    lane = T.cuda.lane_id()
    tile = T.alloc_shared(({size},), "float32", align=128)
    barrier = T.alloc_shared((1,), "uint64")
    for j in T.serial({size // 32}):
        tile[j * 32 + lane] = T.cast(j * 32 + lane + 1, "float32")
    if lane == 0:
        T.ptx.mbarrier.init.shared.b64(barrier.ptr_to([0]), 1)
    T.ptx.fence.proxy.async_.shared__cta()
    T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.cta_sync()
    if lane == 0:
        T.ptx["{spelling}"]({", ".join(operands)})
{completion}
    T.cuda.cta_sync()
    for j in T.serial({size // 32}):
        out[j * 32 + lane] = tile[j * 32 + lane]
""",
        {"T": T},
    )


def clip_inputs(d0, d1):
    """Arguments for ``clip_kernel``: the descriptor's own (64, 64) tensor is
    ignored by the override; ``y`` holds distinct values to catch any write."""

    return {
        "m": dict(
            array=np.full((64, 64), -1, np.float32),
            global_shape=(64, 64),
            global_strides=(256,),
            box_shape=CLIP_BOX,
            element_strides=(1, 1),
        ),
        "y": np.arange(32768 + CLIP_PITCH * 8, dtype=np.float32) + 1000,
        "out": np.zeros(CLIP_BOX[0] * CLIP_BOX[1], np.float32),
        "d0": np.int32(d0),
        "d1": np.int32(d1),
    }


def numsim_clip_inputs(inputs):
    metadata = dict(inputs["m"])
    array = metadata.pop("array")
    return {**inputs, "m": numsim.TensorMap(array, **metadata).numpy()}


def clipped_extents(route, columns, rows, chunked=True):
    """SM107 (VR200) measurement: loads bound both axes per element; stores
    and reductions with overridden dimensions bound outer axes per element
    but write every 16-byte chunk of the inner axis that holds an in-bounds
    element. Plain-descriptor stores (``chunked=False``) are modeled per
    element: VR200 showed 16-byte chunks there too, but that is left
    unmodeled until Blackwell is measured."""

    width, height = CLIP_BOX
    if route != "load" and chunked:
        columns = -(-columns // 4) * 4
    return min(columns, width), min(rows, height)


def clipped_expectation(route, columns, rows, y, pitch=CLIP_PITCH, chunked=True):
    """Expected (global buffer, shared tile) for a box moved at offset zero."""

    width, height = CLIP_BOX
    columns, rows = clipped_extents(route, columns, rows, chunked)
    y = y.copy()
    tile = np.arange(1, width * height + 1, dtype=np.float32).reshape(height, width)
    for row in range(height):
        for column in range(width):
            index = row * pitch + column
            inside = row < rows and column < columns
            if route == "load":
                tile[row, column] = y[index] if inside else 0
            elif inside:
                y[index] = tile[row, column] + (y[index] if route == "reduce" else 0)
    return y, tile.reshape(-1)


# (d0, d1) size-minus-one operands: partial and full 16-byte inner chunks,
# partial rows, both axes at once, and operands beyond the box. (An inner
# size past the 128-byte row pitch is rejected as an overlapping stride.)
CLIP_OPERANDS = ((0, 7), (3, 7), (4, 7), (8, 2), (15, 4), (5, 4), (31, 255))


@pytest.mark.parametrize("route", ("load", "s2g", "reduce"))
@pytest.mark.parametrize("d0,d1", CLIP_OPERANDS)
def test_tma_override_dimensions_bound_every_direction(route, d0, d1, tmp_path):
    inputs = numsim_clip_inputs(clip_inputs(d0, d1))
    expected_y, expected_tile = clipped_expectation(route, d0 + 1, d1 + 1, inputs["y"])
    result = run_checked(clip_kernel(route), inputs, cache_dir=tmp_path)
    np.testing.assert_array_equal(inputs["y"], expected_y)
    np.testing.assert_array_equal(result.outputs["out"], expected_tile)


@pytest.mark.parametrize("route", ("load", "s2g", "reduce"))
@pytest.mark.parametrize("shape", ((5, 8), (9, 3), (16, 8)))
def test_tma_plain_store_inner_bound_stays_per_element(route, shape, tmp_path):
    """Without overrides the descriptor's own dimensions bound every direction
    per element, so nothing lands in the 128-byte row gaps of ``y`` past the
    inner bound (VR200 writes whole 16-byte chunks here; deliberately not
    modeled until Blackwell is measured)."""

    inputs = clip_inputs(0, 0)
    inputs["m"] = numsim.TensorMap(
        inputs["y"][: 8 * CLIP_PITCH].reshape(8, CLIP_PITCH),
        global_shape=shape,
        global_strides=(CLIP_PITCH * 4,),
        box_shape=CLIP_BOX,
        element_strides=(1, 1),
    ).numpy()
    expected, expected_tile = clipped_expectation(route, *shape, inputs["y"], chunked=False)
    result = run_checked(clip_kernel(route, override=False), inputs, cache_dir=tmp_path)
    np.testing.assert_array_equal(inputs["y"], expected)
    np.testing.assert_array_equal(result.outputs["out"], expected_tile)


def packed_ragged_store_kernel(starts, counts):
    """CTA ``i`` stores one 16 x 8 box at row ``starts[i]`` of a packed
    output, bounded by the overridden row count ``counts[i]``."""

    selections = "".join(
        f"""
    if cta == {cta}:
        start[0] = {start}
        count[0] = {count}"""
        for cta, (start, count) in enumerate(zip(starts, counts))
    )
    return tvm_tirx.script.from_source(
        f"""
@T.function
def kernel(m: T.Ptr(T.TensorMap), y: T.Tensor((32768 + 512,), "float32")):
    T.device_entry(launch=T.cuda.LaunchConfig(grid={len(starts)}, block=32))
    cta = T.cuda.block_idx("x")
    lane = T.cuda.lane_id()
    tile = T.alloc_shared((128,), "float32", align=128)
    start = T.alloc_local((1,), "int32")
    count = T.alloc_local((1,), "int32")
    for j in T.serial(4):
        tile[j * 32 + lane] = T.cast(cta + 1, "float32")
    T.ptx.fence.proxy.async_.shared__cta()
    T.cuda.cta_sync()
    start[0] = 0
    count[0] = 1{selections}
    if lane == 0:
        T.ptx["cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group.override::global_address.override::global_dim_stride"](
            T.address_of(m), T.reinterpret(y.ptr_to([start[0] * 16]), ty="uint64"),
            T.cast(15, "uint8"), T.cast(count[0] - 1, "uint8"), T.uint32(4), T.uint16(0),
            T.int32(0), T.int32(0), tile.ptr_to([0]))
        T.ptx.cp.async_.bulk.commit_group()
        T.ptx.cp.async_.bulk.wait_group(0)
""",
        {"T": T},
    )


def packed_ragged_inputs():
    return {
        "m": numsim.TensorMap(
            np.zeros((8, 16), np.float32),
            global_shape=(16, 8),
            global_strides=(64,),
            box_shape=(16, 8),
            element_strides=(1, 1),
        ).numpy(),
        "y": np.full(32768 + 512, -7, np.float32),
    }


def test_tma_override_store_clipping_separates_packed_segments(tmp_path):
    counts = (5, 3, 6)
    inputs = packed_ragged_inputs()
    run_checked(packed_ragged_store_kernel((0, 5, 8), counts), inputs, cache_dir=tmp_path)
    expected = np.full(32768 + 512, -7, np.float32)
    expected[: 16 * sum(counts)] = np.repeat(np.arange(1, 4), np.array(counts) * 16)
    np.testing.assert_array_equal(inputs["y"], expected)


def test_tma_override_store_overlapping_segments_still_race():
    # CTA 1's four rows (5..8) reach CTA 2's first row (8).
    report = racecheck(packed_ragged_store_kernel((0, 5, 8), (5, 4, 6)), packed_ragged_inputs())
    assert report.verdict == "error", report.format()
    assert any(finding.details["access_pair"] == "write_write" for finding in report.findings)
