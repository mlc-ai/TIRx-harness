"""Host-encoded atomicity: exact shared permutation and untouched boundaries."""

import numpy as np
import tvm
from tvm.script import tirx as T

from tests.numsim.runtime.test_tma_u6 import u6_case
from tests.numsim.support.execution import assert_rejected, run_checked
from tirx_harness import numsim


def atomicity_case(
    atom, *, flip=False, store=False, im2col=False, shared_base=128, restore_swizzle=False,
):
    if flip and (atom != 32 or store):
        raise ValueError("8B flip is a 32B-atomicity load-only mode")
    swizzle = "128B" if atom == 16 else f"128B_ATOM_{atom}B"
    if flip:
        swizzle += "_FLIP_8B"
    rank = 3 if im2col else 2
    coords = ", ".join(["0"] * rank)
    values = np.arange(128, dtype=np.uint32) + 0x10000000
    base = np.full(132, 0xA5B6C7D8, np.uint32) if store else values
    inputs = {"values": values, "backing": base} if store else {"output": np.zeros(164, np.uint32)}
    # PTX swizzle table: XOR the row index into the atomic-group column.
    # Nonzero shared bases exercise the invocation's starting phase.
    address = f"{shared_base} + i * 4"
    address = f"({address}) ^ (((({address}) >> 7) & {128 // atom - 1}) * {atom})"
    if flip:
        address = f"({address}) ^ ((({shared_base} + i * 4) >> 4) & 8)"
    if store:
        body = f"""
    for i in T.serial(128):
        if lane == 0:
            shared[({address}) // 4] = values[i]
    T.ptx.fence.proxy.async_.shared__cta()
    T.cuda.cta_sync()
    if lane == 0:
        T.ptx["cp.async.bulk.tensor.{rank}d.global.shared::cta.{"im2col_no_offs" if im2col else "tile"}.bulk_group"](
            T.address_of(descriptor), {coords}, shared.ptr_to([{shared_base // 4}]))
        T.ptx.cp.async_.bulk.commit_group()
        T.ptx.cp.async_.bulk.wait_group(0)
"""
        parameters = 'values: T.Buffer((128,), "uint32"), backing: T.Buffer((132,), "uint32")'
        expected = np.concatenate((values, base[128:]))
    else:
        body = f"""
    barrier = T.alloc_shared((1,), "uint64")
    if lane == 0:
        for i in T.serial(164):
            shared[i] = T.uint32(0xa5b6c7d8)
        T.ptx.mbarrier.init.shared.b64(barrier.ptr_to([0]), 1)
    T.ptx.fence.proxy.async_.shared__cta()
    T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.cta_sync()
    if lane == 0:
        T.ptx.mbarrier.arrive.expect_tx.shared.b64(barrier.ptr_to([0]), 512)
        T.ptx["cp.async.bulk.tensor.{rank}d.shared::cta.global.{"im2col." if im2col else ""}mbarrier::complete_tx::bytes"](
            shared.ptr_to([{shared_base // 4}]), T.address_of(descriptor), {coords}, barrier.ptr_to([0]){", T.uint16(0)" if im2col else ""})
        T.cuda.mbarrier_wait(barrier.ptr_to([0]), 0)
        for i in T.serial(164):
            output[i] = shared[i]
"""
        parameters = 'output: T.Buffer((164,), "uint32")'
        expected = np.full(164, 0xA5B6C7D8, np.uint32)
        for i, value in enumerate(values):
            physical = shared_base + i * 4
            physical ^= ((physical // 128) % (128 // atom)) * atom
            if flip and (i // 32 + shared_base // 128) % 2:
                physical ^= 8
            expected[physical // 4] = value
    descriptor_type = 'T.TensorMap()'
    if restore_swizzle:
        descriptor_type = 'T.Buffer((128,), "uint8")'
        body = body.replace("T.address_of(descriptor)", "descriptor.ptr_to([0])")
        body = '''
    if lane == 0:
        T.ptx["tensormap_replace.tile.swizzle_mode.global.b1024.b32"](descriptor.ptr_to([0]), 3)
        T.ptx.fence.proxy.tensormap__generic.release.gpu()
        T.ptx.fence.proxy.tensormap__generic.acquire.gpu(descriptor.ptr_to([0]))
''' + body
    kernel = tvm.script.from_source(
        f"""@T.prim_func
def kernel(descriptor: {descriptor_type}, {parameters}):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_shared((164,), "uint32", align=1024)
    {body}
""",
        {"T": T},
    )
    metadata = dict(
        global_shape=(32, 4, 1) if im2col else (32, 4),
        global_strides=(128, 512) if im2col else (128,),
        box_shape=(32, 4),
        element_strides=(1,) * rank,
        swizzle=swizzle,
        im2col=numsim.Im2col((0,), (0,)) if im2col else None,
    )
    return kernel, inputs, base, metadata, expected


ATOMICITY_CASES = (
    # Keep every layout/direction, with representative descriptor restoration.
    # atom, flip, store, im2col, shared_base, restore_swizzle
    (32, False, False, False, 0, False),
    (32, False, True, False, 0, True),
    (32, True, False, False, 128, True),
    (64, False, True, False, 0, True),
    (32, False, False, True, 128, True),
    (32, False, True, True, 128, False),
    (32, True, False, True, 128, False),
    (64, False, True, True, 128, False),
)


def test_tma_atomicity_layout_and_checkers(tmp_path):
    for atom, flip, store, im2col, shared_base, restore_swizzle in ATOMICITY_CASES:
        kernel, inputs, base, metadata, expected = atomicity_case(
            atom,
            flip=flip,
            store=store,
            im2col=im2col,
            shared_base=shared_base,
            restore_swizzle=restore_swizzle,
        )
        inputs["descriptor"] = numsim.TensorMap(base, **metadata).numpy()
        if restore_swizzle:
            inputs["descriptor"][60] &= 0xB3  # Disable width, retaining the atomicity field.
        output = "backing" if store else "output"
        actual = run_checked(kernel, inputs, cache_dir=tmp_path, outputs=(output,)).outputs[output]
        np.testing.assert_array_equal(actual, expected)


def test_tma_atomicity_direction_restrictions(tmp_path):
    for store, swizzle, dtype, verdict, message in (
        (True, "128B_ATOM_32B_FLIP_8B", None, "error", "8B flip is only valid for global-to-shared"),
        (False, "128B_ATOM_64B", None, "incomplete", "tma_64b_atomicity_load_unmodeled"),
        (False, "128B_ATOM_64B", "uint6", "error", "64B atomicity loads are invalid for U6 and padded FP4"),
        (False, "128B_ATOM_64B", "fp4", "error", "64B atomicity loads are invalid for U6 and padded FP4"),
    ):
        if dtype is None:
            kernel, inputs, base, metadata, _ = atomicity_case(32, store=store)
        else:
            kernel, inputs, base, metadata, _ = u6_case(swizzle=True, atomicity=32)
            if dtype == "fp4":
                metadata.update(tma_dtype=None, fp4_shared_layout="align16_padded", global_strides=(64,))
        inputs["descriptor"] = numsim.TensorMap(base, **{**metadata, "swizzle": swizzle}).numpy()
        for report in assert_rejected(kernel, inputs, message, verdict=verdict, cache_dir=tmp_path):
            assert len(report.findings) == 1, report.format()
            operation = report.findings[0].details["operation"]
            source = operation["source"]
            assert operation["source_op_id"] == source["source_op_id"]
            assert "T.ptx.cp(" in source["source_text"]
            assert source["source_span"]["line"] > 0
