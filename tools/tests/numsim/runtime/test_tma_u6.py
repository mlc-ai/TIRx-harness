"""SM100 U6 TMA: packed loads and byte-per-value stores are not inverses."""

import numpy as np
import tvm
from tvm.script import tirx as T

from tests.numsim.support.execution import run_checked
from tirx_harness import numsim


U6_CASES = (
    # Each direction covers plain/swizzled and tile/im2col, without their product.
    (False, False, False, 16), (False, True, True, 16),
    (True, True, False, 16), (True, False, True, 16),
    (False, True, False, 32), (True, True, True, 32), (True, True, False, 64),
)


def u6_case(*, store=False, swizzle=False, im2col=False, atomicity=16):
    rank = 3 if im2col else 2
    coordinates = ", ".join(["0"] * rank)
    load_mode = "im2col." if im2col else ""
    store_mode = "im2col_no_offs" if im2col else "tile"
    offset = ", T.uint16(0)" if im2col else ""
    address = "(i // 3) * 16 + (i % 3) * 4"
    if swizzle:
        address = f"({address}) ^ (((({address}) >> 7) & {128 // atomicity - 1}) * {atomicity})"
    source = (np.arange(256, dtype=np.uint32) * 37 + 13).astype(np.uint8)
    if not store:
        body = f"""
    shared = T.alloc_shared((68,), "uint32", align=1024)
    barrier = T.alloc_shared((1,), "uint64")
    if lane == 0:
        for i in T.serial(4):
            shared[64 + i] = T.uint32(0xa5b6c7d8)
        T.ptx.mbarrier.init.shared.b64(barrier.ptr_to([0]), 1)
    T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.cta_sync()
    if lane == 0:
        T.ptx.mbarrier.arrive.expect_tx.shared.b64(barrier.ptr_to([0]), 192)
        T.ptx["cp.async.bulk.tensor.{rank}d.shared::cta.global.{load_mode}mbarrier::complete_tx::bytes"](
            shared.ptr_to([0]), T.address_of(descriptor), {coordinates}, barrier.ptr_to([0]){offset})
        T.cuda.mbarrier_wait(barrier.ptr_to([0]), 0)
        for i in T.serial(48):
            output[i] = shared[({address}) // 4]
        for i in T.serial(4):
            output[48 + i] = shared[64 + i]
"""
        parameters = 'output: T.Buffer((52,), "uint32")'
        inputs = {"output": np.zeros(52, np.uint32)}
        # Do not read the unspecified four padding bytes of each 16-byte atom.
        expected = np.concatenate((source[:192].view(np.uint32), np.full(4, 0xA5B6C7D8, np.uint32)))
    else:
        source.fill(0xA5)
        values = (((np.arange(256, dtype=np.uint32) * 37 + 13) & 63) | 192).astype(np.uint8)
        address = "i * 32 + lane"
        if swizzle:
            address = f"({address}) ^ (((({address}) >> 7) & {128 // atomicity - 1}) * {atomicity})"
        body = f"""
    shared = T.alloc_shared((256,), "uint8", align=1024)
    for i in T.serial(8):
        shared[{address}] = values[i * 32 + lane]
    T.ptx.fence.proxy.async_.shared__cta()
    T.cuda.cta_sync()
    if lane == 0:
        T.ptx["cp.async.bulk.tensor.{rank}d.global.shared::cta.{store_mode}.bulk_group"](
            T.address_of(descriptor), {coordinates}, shared.ptr_to([0]))
        T.ptx.cp.async_.bulk.commit_group()
        T.ptx.cp.async_.bulk.wait_group(0)
"""
        parameters = 'values: T.Buffer((256,), "uint8"), backing: T.Buffer((256,), "uint8")'
        inputs = {"values": values}
        # Independent bitstream oracle; high bits are deliberately nonzero.
        bits = np.unpackbits(values[:, None], axis=1, bitorder="little")[:, :6].reshape(-1)
        expected = np.concatenate((np.packbits(bits, bitorder="little"), source[192:]))
    kernel = tvm.script.from_source(
        f"""@T.prim_func
def kernel(descriptor: T.TensorMap(), {parameters}):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    {body}
""",
        {"T": T},
    )
    storage = np.empty(source.size + 31, np.uint8)
    start = -int(storage.ctypes.data) % 32
    base = storage[start : start + source.size]
    base[:] = source
    if store:
        inputs["backing"] = base
    metadata = dict(
        global_shape=(128, 2, 1) if im2col else (128, 2),
        global_strides=(96, 192) if im2col else (96,),
        box_shape=(128, 2),
        element_strides=(1,) * rank,
        tma_dtype="uint6",
        swizzle=("128B" if atomicity == 16 else f"128B_ATOM_{atomicity}B") if swizzle else None,
        im2col=numsim.Im2col((0,), (0,)) if im2col else None,
    )
    return kernel, inputs, base, metadata, expected


def test_tma_u6_layout_and_checkers(tmp_path):
    for store, swizzle, im2col, atomicity in U6_CASES:
        kernel, inputs, base, metadata, expected = u6_case(
            store=store,
            swizzle=swizzle,
            im2col=im2col,
            atomicity=atomicity,
        )
        inputs["descriptor"] = numsim.TensorMap(base, **metadata).numpy()
        output = "backing" if store else "output"
        actual = run_checked(kernel, inputs, cache_dir=tmp_path, outputs=(output,)).outputs[output]
        np.testing.assert_array_equal(actual, expected)
