from __future__ import annotations

import numpy as np

from tirx_harness import numsim
from tvm_tirx.script import tirx as T

# Layout F (M=64) writes TMEM lanes 0..15 of each 32-lane quarter; row r lives in
# lane (r // 16) * 32 + r % 16. The 16x32bx2 shape addresses exactly those 16 lanes
# (or lanes 16..31 with a TLane base of 16), so each warp initializes its quarter.
_HOLE_LANES = 16 << 16


@T.function
def raw_tf32_layout_f_hole_poison(
    consume_holes: T.int32,
    a: T.Tensor((64, 8), "float32"),
    b_physical: T.Tensor((4096,), "uint8"),
    valid: T.Tensor((64, 32), "float32"),
    holes: T.Tensor((4, 16), "uint32"),
):
    T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=128))
    warp = T.cuda.warp_id()
    lane = T.cuda.lane_id()
    shared_b = T.alloc_shared((4096,), "uint8")
    address = T.alloc_shared((1,), "uint32")
    registers = T.alloc_local((16,), "uint32")
    readback = T.alloc_local((2,), "uint32")
    desc_i: T.uint32
    desc_b: T.uint64

    if warp == 0:
        T.ptx.tcgen05.alloc.cta_group__1.sync.aligned.shared__cta.b32(T.address_of(address[0]), 64)
    thread = warp * 32 + lane
    for copy_i in T.serial(32):
        offset = thread + copy_i * 128
        shared_b[offset] = b_physical[offset]
    T.cuda.cta_sync()

    # A in columns 0..7; the upper half-warp's split chunk lands in unused columns 56..63.
    row = warp * 16 + lane % 16
    for k in T.unroll(8):
        registers[k] = T.reinterpret(a[row, k], ty="uint32")
    T.ptx["tcgen05.st.sync.aligned.16x32bx2.x8.b32"](
        address[0],
        56,
        registers[0],
        registers[1],
        registers[2],
        registers[3],
        registers[4],
        registers[5],
        registers[6],
        registers[7],
    )
    # The D accumulator: columns 16..31 from the lower half-warp, 32..47 from the upper.
    for col in T.unroll(16):
        registers[col] = T.reinterpret(T.float32(4), ty="uint32")
    T.ptx["tcgen05.st.sync.aligned.16x32bx2.x16.b32"](
        address[0] + T.uint32(16),
        16,
        registers[0],
        registers[1],
        registers[2],
        registers[3],
        registers[4],
        registers[5],
        registers[6],
        registers[7],
        registers[8],
        registers[9],
        registers[10],
        registers[11],
        registers[12],
        registers[13],
        registers[14],
        registers[15],
    )
    # Holes: lanes 16..31 of column 16 hold a sentinel; their column 17 stays unwritten.
    registers[0] = T.uint32(0x3F400000)
    T.ptx["tcgen05.st.sync.aligned.16x32bx2.x1.b32"](
        address[0] + T.uint32(_HOLE_LANES + 16), 40, registers[0]
    )
    T.ptx.tcgen05.wait__st.sync.aligned()
    T.ptx.tcgen05.fence__before_thread_sync()
    T.cuda.cta_sync()
    T.ptx.tcgen05.fence__after_thread_sync()

    if warp == 0:
        T.cuda.tcgen05.encode_instr_descriptor(
            T.address_of(desc_i),
            d_dtype="float32",
            a_dtype="tf32",
            b_dtype="tf32",
            M=64,
            N=32,
            K=8,
            trans_a=False,
            trans_b=False,
            n_cta_groups=1,
        )
        T.cuda.tcgen05.encode_matrix_descriptor(
            T.address_of(desc_b), T.address_of(shared_b[0]), ldo=0, sdo=64, swizzle=3
        )
        T.ptx["tcgen05.mma.cta_group::1.kind::tf32"](
            address[0] + T.uint32(16),
            address[0],
            desc_b,
            desc_i,
            T.uint32(0),
            T.uint32(0),
            T.uint32(0),
            T.uint32(0),
            T.ptx.pred(T.uint32(1)),
            pred=T.cast(lane == 7, "uint32"),
        )
    T.cuda.cta_sync()
    T.ptx.tcgen05.fence__after_thread_sync()
    T.ptx["tcgen05.ld.sync.aligned.32x32b.x1.b32"](
        readback[0], T.cuda.get_tmem_addr(address[0], 0, 16)
    )
    T.ptx["tcgen05.ld.sync.aligned.32x32b.x1.b32"](
        readback[1], T.cuda.get_tmem_addr(address[0], 0, 17)
    )
    T.ptx["tcgen05.ld.sync.aligned.16x32bx2.x16.b32"](
        registers[0],
        registers[1],
        registers[2],
        registers[3],
        registers[4],
        registers[5],
        registers[6],
        registers[7],
        registers[8],
        registers[9],
        registers[10],
        registers[11],
        registers[12],
        registers[13],
        registers[14],
        registers[15],
        address[0] + T.uint32(16),
        16,
    )
    T.ptx.tcgen05.wait__ld.sync.aligned()

    for col in T.unroll(16):
        valid[row, (lane // 16) * 16 + col] = T.reinterpret(registers[col], ty="float32")
    if consume_holes != 0:
        if lane >= 24:
            holes[warp, lane - 16] = readback[1]
        elif lane >= 16:
            holes[warp, lane - 16] = readback[0]
    T.cuda.cta_sync()
    if warp == 0:
        T.ptx.tcgen05.dealloc.cta_group__1.sync.aligned.b32(address[0], 64)
        T.ptx.tcgen05.relinquish_alloc_permit.cta_group__1.sync.aligned()


def _kmajor_swizzle_physical(logical_bytes: np.ndarray) -> np.ndarray:
    rows, row_bytes = logical_bytes.shape
    swizzle_len = 3
    stride_bytes = 1024
    physical = np.zeros(((rows + 7) // 8) * stride_bytes, dtype=np.uint8)
    row_stride = 16 << swizzle_len
    swizzle_mask = (1 << swizzle_len) - 1
    for row in range(rows):
        for byte_in_row in range(row_bytes):
            atom = byte_in_row // 16
            byte_in_atom = byte_in_row % 16
            unswizzled = (
                (row % 8) * row_stride + (row // 8) * stride_bytes + atom * 16 + byte_in_atom
            )
            atom_index = unswizzled >> 4
            swizzled_atom = atom_index ^ (
                (atom_index & (swizzle_mask << swizzle_len)) >> swizzle_len
            )
            physical[(swizzled_atom << 4) | byte_in_atom] = logical_bytes[row, byte_in_row]
    return physical


def test_tf32_layout_f_unwritten_holes_are_zero_filled_and_require_review(tmp_path):
    row = np.arange(64, dtype=np.float32)[:, None]
    k = np.arange(8, dtype=np.float32)[None, :]
    col = np.arange(32, dtype=np.float32)[:, None]
    a = ((row % np.float32(5)) - np.float32(2)) * np.float32(0.5) + k * np.float32(0.25)
    b = ((col % np.float32(7)) - np.float32(3)) * np.float32(0.25) - k * np.float32(0.5)
    b_bytes = np.ascontiguousarray(b).view(np.uint8).reshape(32, 32)
    valid = np.zeros((64, 32), dtype=np.float32)
    holes = np.zeros((4, 16), dtype=np.uint32)

    module = numsim.transpile(raw_tf32_layout_f_hole_poison, cache_dir=tmp_path)
    result = numsim.Engine().run(
        module,
        {
            "consume_holes": np.int32(1),
            "a": np.ascontiguousarray(a),
            "b_physical": _kmajor_swizzle_physical(b_bytes),
            "valid": valid,
            "holes": holes,
        },
    )

    expected_holes = np.zeros_like(holes)
    expected_holes[:, :8] = np.uint32(0x3F400000)
    np.testing.assert_array_equal(result.outputs["holes"], expected_holes)
    assert result.verdict == "review"
    assert {item["kind"] for item in result.diagnostics} == {"uninitialized_read"}
