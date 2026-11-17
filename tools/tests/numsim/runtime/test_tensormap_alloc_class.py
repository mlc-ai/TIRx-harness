"""TensorMap allocation class: tensormap.replace .alloc_class and tensormap.query.

PTX ISA 9.5 (9.7.10.29/30): the class is small or large; the field-less form
sets it, ``global_address`` takes ``.alloc_small``/``.alloc_large``/
``.alloc_same`` and resets it to small without a qualifier. The query reports
whether it is large. PTX does not say which class host encoding picks; on
VR200 (CUDA 13.5) cuTensorMapEncode* picks large exactly when the encoded
tensor span reaches 128 KiB, which NumSim reproduces.
"""

import numpy as np
import pytest

import tvm_tirx
from tests.numsim.support.execution import run_checked
from tirx_harness import numsim
from tvm_tirx.script import tirx as T

LANES = 4

# (form, qualifier, lanes that issue it). "class" is the field-less form;
# "address" is global_address, without a qualifier when it is None.
STEPS = (
    ("class", "alloc_large", 0b1111),
    ("class", "alloc_small", 0b0101),
    ("address", None, 0b0011),
    ("address", "alloc_large", 0b0100),
    ("address", "alloc_same", 0b1111),
    ("class", "alloc_large", 0b0001),
    ("address", "alloc_same", 0b0011),
    ("address", "alloc_small", 0b1000),
    ("class", "alloc_large", 0b1010),
    ("address", None, 0b0010),
    ("address", "alloc_same", 0b1100),
)

# Descriptor storage and replacement space, then the query's address space.
SPACES = (
    ("global", "global"),
    ("global", ""),
    ("", "global"),
    ("shared::cta", ""),
)


def expected_queries():
    """Independent replay of the PTX class rules for each lane."""
    # The (4, 2) uint32 host maps span 32 bytes: small.
    classes = ["alloc_small"] * LANES
    rows = []
    for form, qualifier, lanes in STEPS:
        for lane in range(LANES):
            if not lanes >> lane & 1:
                continue
            if form == "class":
                classes[lane] = qualifier
            elif qualifier is None:
                classes[lane] = "alloc_small"
            elif qualifier != "alloc_same":
                classes[lane] = qualifier
        rows.append([int(value == "alloc_large") for value in classes])
    return np.array(rows, dtype=np.uint32)


def _pointer(base, space):
    # Lanes past LANES are predicated off; keep their addresses inside the
    # objects anyway. With lane * 128, the shared/generic variant hit an
    # illegal memory access on SM107 (lanes 4..31 point past the 512-byte
    # shared image).
    pointer = f"{base}.ptr_to([lane % {LANES} * 128])"
    if not space:
        pointer = f'T.reinterpret(T.reinterpret({pointer}, ty="uint64"), ty="handle")'
    return pointer


def _replace(form, qualifier, space, pointer, predicate):
    suffix = f"{space}.b1024" if space else "b1024"
    if form == "class":
        return f'T.ptx["tensormap_replace.tile.{qualifier}.{suffix}"]({pointer}, pred={predicate})'
    qualifier = f"{qualifier}." if qualifier else ""
    return (
        f'T.ptx["tensormap_replace.tile.global_address.{qualifier}{suffix}.b64"]('
        f'{pointer}, T.reinterpret(replacement.ptr_to([lane % {LANES} * 4]), ty="uint64"), '
        f"pred={predicate})"
    )


def query_kernel(replace_space, query_space, steps=STEPS):
    shared = replace_space == "shared::cta"
    image = "image" if shared else "descriptors"
    replace_pointer = _pointer(image, replace_space)
    query_pointer = _pointer(image, query_space)
    query_suffix = f"{query_space}.b1024" if query_space else "b1024"
    lines = []
    for index, (form, qualifier, lanes) in enumerate(steps):
        predicate = f"lane < {LANES} and ({lanes} >> lane) & 1 != 0"
        lines.append("    " + _replace(form, qualifier, replace_space, replace_pointer, predicate))
        lines.append(
            f'    T.ptx["tensormap_query.alloc_class_large.{query_suffix}"]('
            f"output[{index}, lane], {query_pointer}, pred=lane < {LANES})"
        )
    body = "\n".join(lines)
    return tvm_tirx.script.from_source(
        f"""
@T.function
def kernel(descriptors: T.Tensor(({LANES * 128},), "uint8"), replacement: T.Tensor((16,), "uint32"),
           output: T.Tensor(({len(steps)}, 32), "uint32")):
    T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=32))
    lane = T.cuda.lane_id()
    image = T.alloc_shared(({LANES * 128},), "uint8", align=128)
    for i in T.serial({LANES * 4}):
        image[i * 32 + lane] = descriptors[i * 32 + lane]
    T.cuda.cta_sync()
{body}
""",
        {"T": T},
    )


def _host_descriptors():
    source = np.arange(1, 9, dtype=np.uint32)
    descriptor = numsim.TensorMap(
        base=source,
        global_shape=(4, 2),
        global_strides=(16,),
        box_shape=(4, 1),
        element_strides=(1, 1),
    ).numpy()
    return np.tile(descriptor, LANES)


def _query_inputs(steps):
    return {
        "descriptors": _host_descriptors(),
        "replacement": np.arange(41, 57, dtype=np.uint32),
        "output": np.full((len(steps), 32), 0xDEADBEEF, np.uint32),
    }


@pytest.mark.parametrize(("replace_space", "query_space"), SPACES)
def test_query_reports_class_after_each_replace(replace_space, query_space):
    kernel = query_kernel(replace_space, query_space)
    result = run_checked(kernel, _query_inputs(STEPS), outputs=("output",))
    np.testing.assert_array_equal(result.outputs["output"][:, :LANES], expected_queries())


def predicated_off_query_kernel(space, stride, offset=0):
    """Lanes 0..3 query valid objects; predicated-off lanes 4..31 address
    ``lane * stride + offset`` bytes into the same 512-byte buffer."""
    target = "image" if space == "shared" else "descriptors"
    pointer = (
        f'T.reinterpret(T.reinterpret({target}.ptr_to([0]), ty="uint64") + '
        f"T.uint64(T.if_then_else(lane < {LANES}, lane * 128, lane * {stride} + {offset})), "
        'ty="handle")'
    )
    return tvm_tirx.script.from_source(
        f"""
@T.function
def kernel(descriptors: T.Tensor(({LANES * 128},), "uint8"), output: T.Tensor((32,), "uint32")):
    T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=32))
    lane = T.cuda.lane_id()
    image = T.alloc_shared(({LANES * 128},), "uint8", align=128)
    for i in T.serial({LANES * 4}):
        image[i * 32 + lane] = descriptors[i * 32 + lane]
    T.cuda.cta_sync()
    T.ptx["tensormap_query.alloc_class_large.b1024"](output[lane], {pointer}, pred=lane < {LANES})
""",
        {"T": T},
    )


def _predicated_off_inputs():
    return {"descriptors": _host_descriptors(), "output": np.full(32, 0xDEADBEEF, np.uint32)}


@pytest.mark.parametrize("space", ("shared", "global"))
def test_predicated_off_query_lanes_need_valid_descriptors(space):
    # SM107: every lane reaching `@p tensormap.query` must address a valid
    # descriptor, even with p false (unlike `@p ld` / `@p tensormap.replace`).
    for stride, offset in ((128, 0), (0, 64)):
        kernel = predicated_off_query_kernel(space, stride, offset)
        # A generic address past the shared allocation already fails when
        # NumSim resolves it to a shared view, before the descriptor check.
        reason = "faults on SM107|raw shared address .* exceeds allocation"
        with pytest.raises(numsim.NumSimExecutionError, match=reason):
            numsim.Engine().run(numsim.transpile(kernel), _predicated_off_inputs())
    # In-range predicated-off lanes are fine and only issuing lanes report.
    result = run_checked(
        predicated_off_query_kernel(space, 0), _predicated_off_inputs(), outputs=("output",)
    )
    np.testing.assert_array_equal(result.outputs["output"][:LANES], [0] * LANES)


# (elements, row stride in bytes, rows) of uint32 host maps, and their
# measured default class: large exactly when the span reaches 128 KiB.
HOST_MAPS = {
    "131056_bytes": ((32764,), (), 0),
    "131072_bytes": ((32768,), (), 1),
    "wide_stride": ((4, 2), (1 << 20,), 1),
}


def host_query_kernel(elements, alloc_same):
    replace = (
        '        T.ptx["tensormap_replace.tile.global_address.alloc_same.global.b1024.b64"]('
        'descriptor.ptr_to([0]), T.reinterpret(base.ptr_to([0]), ty="uint64"))\n'
        if alloc_same
        else ""
    )
    return tvm_tirx.script.from_source(
        f"""
@T.function
def kernel(descriptor: T.Tensor((128,), "uint8"), base: T.Tensor(({elements},), "uint32"),
           output: T.Tensor((2,), "uint32")):
    T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=32))
    lane = T.cuda.lane_id()
    if lane == 0:
{replace}        T.ptx["tensormap_query.alloc_class_large.global.b1024"](output[0], descriptor.ptr_to([0]))
        T.ptx["tensormap_query.alloc_class_large.b1024"](
            output[1], T.reinterpret(T.reinterpret(descriptor.ptr_to([0]), ty="uint64"), ty="handle"))
""",
        {"T": T},
    )


def host_map(case):
    """Kernel base length and TensorMap metadata of one HOST_MAPS case."""
    shape, strides, large = HOST_MAPS[case]
    elements = shape[0] if len(shape) == 1 else strides[0] // 4 * (shape[1] - 1) + shape[0]
    metadata = dict(
        global_shape=shape,
        global_strides=strides,
        box_shape=(4,) if len(shape) == 1 else (4, 1),
        element_strides=(1,) * len(shape),
    )
    return elements, metadata, large


@pytest.mark.parametrize("alloc_same", (False, True), ids=("host", "alloc_same"))
@pytest.mark.parametrize("case", HOST_MAPS)
def test_host_encoded_default_class(case, alloc_same):
    elements, metadata, large = host_map(case)
    base = np.zeros(elements, np.uint32)
    result = run_checked(
        host_query_kernel(elements, alloc_same),
        {
            "descriptor": numsim.TensorMap(base=base, **metadata).numpy(),
            "base": base,
            "output": np.full(2, 0xDEADBEEF, np.uint32),
        },
        outputs=("output",),
    )
    np.testing.assert_array_equal(result.outputs["output"], [large, large])


def tma_kernel(updates, row, count=16):
    """Replace the address of a shared copy, publish it, and TMA-load a
    (4, 2) box starting at ``row``; rows outside 0..1 are out of bounds."""
    updates = "\n".join(f"        {update}" for update in updates)
    return tvm_tirx.script.from_source(
        f"""
@T.function
def kernel(descriptor: T.Tensor((128,), "uint8"), destination: T.Tensor((128,), "uint8"),
           replacement: T.Tensor(({count},), "uint32"), output: T.Tensor((8,), "uint32"),
           large: T.Tensor((1,), "uint32")):
    T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=32))
    lane = T.cuda.lane_id()
    image = T.alloc_shared((128,), "uint8", align=128)
    data = T.alloc_shared((8,), "uint32", align=128)
    barrier = T.alloc_shared((1,), "uint64")
    for i in T.serial(4):
        image[i * 32 + lane] = descriptor[i * 32 + lane]
    T.cuda.cta_sync()
    if lane == 0:
{updates}
    T.cuda.warp_sync()
    T.ptx["tensormap_cp_fenceproxy.global.shared::cta.tensormap::generic.release.gpu.sync.aligned"](
        destination.ptr_to([0]), image.ptr_to([0]))
    T.ptx.fence.proxy.tensormap__generic.acquire.gpu(destination.ptr_to([0]), pred=lane == 0)
    if lane == 0:
        T.ptx["tensormap_query.alloc_class_large.global.b1024"](large[0], destination.ptr_to([0]))
        T.ptx.mbarrier.init.shared.b64(barrier.ptr_to([0]), 1)
    T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.cta_sync()
    if lane == 0:
        T.ptx.mbarrier.arrive.expect_tx.shared.b64(barrier.ptr_to([0]), 32)
        T.ptx["cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes"](
            data.ptr_to([0]), destination.ptr_to([0]), 0, {row}, barrier.ptr_to([0]))
        T.cuda.mbarrier_wait(barrier.ptr_to([0]), 0)
    T.cuda.cta_sync()
    if lane < 8:
        output[lane] = data[lane]
""",
        {"T": T},
    )


def _address(qualifier):
    qualifier = f"{qualifier}." if qualifier else ""
    return (
        f'T.ptx["tensormap_replace.tile.global_address.{qualifier}shared::cta.b1024.b64"]('
        'image.ptr_to([0]), T.reinterpret(replacement.ptr_to([0]), ty="uint64"))'
    )


def _class(qualifier):
    return f'T.ptx["tensormap_replace.tile.{qualifier}.shared::cta.b1024"](image.ptr_to([0]))'


# Update sequences with their resulting class (1 = large).
TMA_CASES = {
    "reset": ([_class("alloc_large"), _address(None)], 0),
    "small": ([_address("alloc_small")], 0),
    "large": ([_address("alloc_large")], 1),
    "same_large": ([_class("alloc_large"), _address("alloc_same")], 1),
    "class_after_address": ([_address(None), _class("alloc_large")], 1),
}


TMA_MAP = dict(global_shape=(4, 2), global_strides=(16,), box_shape=(4, 2), element_strides=(1, 1))


def _run_tma(updates, row, run=run_checked, count=16):
    replacement = np.arange(41, 41 + count, dtype=np.uint32)
    result = run(
        tma_kernel(updates, row, count),
        {
            "descriptor": numsim.TensorMap(
                base=np.arange(1, 9, dtype=np.uint32), **TMA_MAP
            ).numpy(),
            "destination": np.zeros(128, np.uint8),
            "replacement": replacement,
            "output": np.full(8, 0xDEADBEEF, np.uint32),
            "large": np.full(1, 0xDEADBEEF, np.uint32),
        },
        outputs=("output", "large"),
    )
    return replacement, result.outputs


@pytest.mark.parametrize("case", TMA_CASES)
def test_tma_copy_is_unchanged_by_class(case):
    updates, large = TMA_CASES[case]
    replacement, outputs = _run_tma(updates, 0)
    # Both rows of the replaced (4, 2) u32 tensor with a 16-byte row stride.
    np.testing.assert_array_equal(outputs["output"], replacement[:8])
    np.testing.assert_array_equal(outputs["large"], [large])


def _zero_filled(replacement, row):
    rows = [replacement[4 * r : 4 * r + 4] if r in (0, 1) else [0] * 4 for r in (row, row + 1)]
    return np.concatenate(rows).astype(np.uint32)


@pytest.mark.parametrize("row", (1, -1))
@pytest.mark.parametrize("case", ("reset", "small"))
def test_small_class_out_of_bounds_tma_zero_fills(case, row):
    replacement, outputs = _run_tma(TMA_CASES[case][0], row)
    np.testing.assert_array_equal(outputs["output"], _zero_filled(replacement, row))


@pytest.mark.parametrize("case", ("large", "same_large", "class_after_address"))
def test_large_class_out_of_bounds_tma_needs_min_size_allocation(case):
    # PTX: undefined unless LARGE_ALLOC_MIN_SIZE (128 KiB in NumSim) bytes
    # from the global address are accessible.
    with pytest.raises(numsim.NumSimExecutionError, match="large allocation-class"):
        _run_tma(
            TMA_CASES[case][0],
            1,
            run=lambda kernel, inputs, outputs: numsim.Engine().run(
                numsim.transpile(kernel), inputs, outputs=outputs
            ),
        )
    count = 128 * 1024 // 4
    replacement, outputs = _run_tma(TMA_CASES[case][0], 1, count=count)
    np.testing.assert_array_equal(outputs["output"], _zero_filled(replacement, 1))
