"""Block-scaled TMEM A uses the existing MMA and scale address contracts."""

import numpy as np
import pytest
from tvm_ffi import structural_map
from tvm.script import tirx as T

from tests.numsim.runtime.test_tcgen05_ti16 import ti16_kernel
from tests.numsim.support.execution import assert_rejected, run_checked
from tirx_harness.numsim.transpiler.ptx_dialect import decode_ptx_call


def block_tmem_case(
    cta_group,
    a_format,
    b_format,
    *,
    block_scale=True,
    tmem_a=True,
    m=None,
    k=32,
    implicit_scale=False,
):
    # Independent encodings for exactly representable values 1, 2, 3, 4.
    encodings = {
        0: (8, [0x38, 0x40, 0x44, 0x48]),
        1: (8, [0x3C, 0x40, 0x42, 0x44]),
        3: (6, [8, 16, 20, 24]),
        4: (6, [12, 16, 18, 20]),
        5: (4, [2, 4, 5, 6]),
    }
    m, n = m or 128 * cta_group, 16 * cta_group
    row, inner = np.indices((m, k))
    index = (row + inner * 3) % 4
    a = (index + 1).astype(np.float32)
    negative = row % 3 == 0
    a[negative] *= -1
    a_width, a_codes = encodings[a_format]
    a_bits = np.array(a_codes, np.uint8)[index] | (negative.astype(np.uint8) << (a_width - 1))
    brow, binner = np.indices((n, k))
    bindex = (brow * 3 + binner) % 4
    b = (bindex + 1).astype(np.float32)
    b_width, b_codes = encodings[b_format]
    b_bits = np.array(b_codes, np.uint8)[bindex]

    def shared_storage(bits, width):
        storage = np.full(bits.shape, 0xA5, np.uint8)
        for row in range(bits.shape[0]):
            for atom in range(k // 16):
                packed = sum(
                    int(value) << (i * width)
                    for i, value in enumerate(bits[row, atom * 16 : (atom + 1) * 16])
                )
                start = atom * (width * 2 if k == 64 else 16)
                storage[row, start : start + width * 2] = list(packed.to_bytes(width * 2, "little"))
        return storage

    b_storage = shared_storage(b_bits, b_width)
    if not tmem_a or k == 64 and block_scale:
        a_bits = shared_storage(a_bits, a_width)
    elif a_format == 5:
        a_bits <<= 2  # PTX padded FP4 TMEM format: 00 S E2 M1 00.
    metadata = np.zeros((2, 128, 2), np.uint32)
    for cta in range(cta_group):
        lane = np.arange(128, dtype=np.uint32)
        metadata[cta, :, 0] = 125 | ((126 + lane % 32 % 3 + cta) << 8) | (129 << 16) | (124 << 24)
        # Both CTAs consume the full B scale vector; each holds a replica,
        # not a distinct scale vector for its local half of shared B.
        metadata[cta, :, 1] = 123 | (129 << 8) | ((127 + lane % 2) << 16) | (126 << 24)
    if block_scale:
        rows = m // cta_group
        a *= np.exp2(((np.arange(m) % rows % 32 % 3) - 1 + np.arange(m) // rows)[:, None])
        b *= np.exp2((np.arange(n) % 2)[:, None])
    seed = np.arange(m * n, dtype=np.float32).reshape(m, n) / 4
    args = {
        "a": np.broadcast_to(a_bits.view(np.uint16), (4, m, k // 2)).copy(),
        "b": b_storage.view(np.uint16),
        "metadata": metadata,
        "zero_mask": np.zeros(1, np.uint64),
        "seed": seed.view(np.int32),
        "out": np.zeros((m, n), np.int32),
    }
    kernel = ti16_kernel(
        True,
        tmem_a,
        cta_group,
        m,
        kind="mxf8f6f4" if block_scale else "f8f6f4",
        a_format=a_format,
        b_format=b_format,
        mma_k=k,
        arch="sm_107a" if k == 64 or implicit_scale else "sm_100a",
        implicit_scale=implicit_scale,
        collectors=".collector::a::discard.collector::b::discard" if implicit_scale else "",
    )
    expected = a @ b.T + seed
    if tmem_a and cta_group == 2 and m == 128:
        # The second N half selects a distinct A bank in each CTA.
        sign = 1 << (5 if a_width < 8 else 7)
        args["a"][1] ^= np.uint16(sign | (sign << 8))
        expected[:, n // 2 :] = (-a) @ b[n // 2 :].T + seed[:, n // 2 :]
    if not block_scale and cta_group == 2:
        columns = n // 2 if m == 128 else n
        expected[m // 2 + 1, :columns] = seed[m // 2 + 1, :columns]
    return kernel, args, expected


@pytest.mark.parametrize(
    "cta_group,a_format,b_format",
    [(1, 0, 1), (2, 1, 3), (1, 3, 4), (2, 4, 5), (1, 5, 0)],
)
def test_block_scaled_tmem_a(cta_group, a_format, b_format, tmp_path):
    kernel, args, expected = block_tmem_case(cta_group, a_format, b_format)
    actual = run_checked(kernel, args, cache_dir=tmp_path).outputs["out"]
    np.testing.assert_array_equal(actual.view(np.float32), expected)

    if (cta_group, a_format, b_format) != (1, 0, 1):
        return
    # Keep the exact K-major control above. Bit 15 alone does not establish
    # a transposed TMEM layout; reserved bit 6 remains an error even with it.
    for extra_bits, verdict, message in (
        (1 << 15, "error", "TMEM A must be K-major"),
        (1 << 6, "error", "valid K/reserved bits"),
        ((1 << 15) | (1 << 6), "error", "valid K/reserved bits"),
    ):

        def replace_descriptor(node):
            if type(node).__name__ != "Call" or not str(node.op.name).startswith(
                "tirx.ptx.tcgen05_mma"
            ):
                return node
            descriptor = decode_ptx_call(node).scalar_operand("idesc")
            operands = [
                T.uint32(int(arg.value) | extra_bits) if arg.same_as(descriptor) else arg
                for arg in node.args
            ]
            return type(node)(
                node.op,
                operands,
                attrs=node.attrs,
                ty_args=node.ty_args,
                span=node.span,
                ret_ty=node.ty,
            )

        changed = kernel.with_body(
            structural_map(kernel.body, replace_descriptor)
        )
        for report in assert_rejected(changed, args, message, verdict=verdict, cache_dir=tmp_path):
            assert "test_tcgen05_ti16.py:" in report.format()


def fp4_tmem_case(tmem_a):
    m, n, k = 256, 32, 128
    codes = np.array([2, 4, 5, 6], np.uint8)
    ar, ak = np.indices((m, k))
    br, bk = np.indices((n, k))
    ai, bi = (ar + ak * 3) % 4, (br * 3 + bk) % 4
    a = (ai + 1).astype(np.float32) * np.where(ar % 3 == 0, -1, 1)
    b = (bi + 1).astype(np.float32)
    a_bits = codes[ai] | ((ar % 3 == 0).astype(np.uint8) << 3)
    b_bits = codes[bi]
    a_storage = (a_bits[:, ::2] | (a_bits[:, 1::2] << 4)).view(np.uint16)
    b_storage = (b_bits[:, ::2] | (b_bits[:, 1::2] << 4)).view(np.uint16)
    metadata = np.zeros((2, 128, 2), np.uint32)
    scale_codes = [126, 127, 128]
    for cta in range(2):
        for lane in range(128):
            for chunk in range(4):
                metadata[cta, lane, 0] |= np.uint32(
                    scale_codes[(lane % 32 + chunk + cta) % 3] << (chunk * 8)
                )
                metadata[cta, lane, 1] |= np.uint32(
                    scale_codes[(lane % 32 + chunk * 2) % 3] << (chunk * 8)
                )
    a *= np.exp2((ar % 32 + (ak // 32) % 4 + ar // 128) % 3 - 1)
    b *= np.exp2((br % 32 + ((bk // 32) % 4) * 2) % 3 - 1)
    seed = np.arange(m * n, dtype=np.float32).reshape(m, n) / 4
    args = {
        "a": np.broadcast_to(a_storage, (4, *a_storage.shape)).copy(),
        "b": b_storage,
        "metadata": metadata,
        "zero_mask": np.zeros(1, np.uint64),
        "seed": seed.view(np.int32),
        "out": np.zeros((m, n), np.int32),
    }
    kernel = ti16_kernel(
        True,
        tmem_a,
        2,
        m,
        kind="mxf4",
        arch="sm_107a",
        mma_k=k,
        implicit_scale=True,
        collectors=".collector::a::discard.collector::b::discard",
    )
    return kernel, args, a @ b.T + seed


@pytest.mark.parametrize("tmem_a", (False, True))
@pytest.mark.parametrize("kind", ("mxf4", "mxf8f6f4"))
def test_block_scale_implicit_size_with_discard(kind, tmem_a, tmp_path):
    if kind == "mxf4":
        kernel, args, expected = fp4_tmem_case(tmem_a)
    else:
        kernel, args, expected = block_tmem_case(
            2, 3 if tmem_a else 0, 1, tmem_a=tmem_a, k=64, implicit_scale=True
        )
    result = run_checked(kernel, args, cache_dir=tmp_path)
    np.testing.assert_array_equal(result.outputs["out"].view(np.float32), expected)
