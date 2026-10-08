"""PTX LUT-B: packed 3-bit indices, per-eight-N lookup tables, full-row reads."""

import numpy as np
import pytest

from tests.numsim.runtime.test_tcgen05_ti16 import ti16_kernel
from tests.numsim.support.execution import run_checked
from tirx_harness import racecheck


def lut_b_case(group=1, tmem=False, segment=0, half=False, block=False, **options):
    m, n, k = 128 * group, 16 * group, 64
    kernel = ti16_kernel(
        True,
        tmem_a=tmem,
        cta_group=group,
        m=m,
        kind="mxf8f6f4" if block else "f8f6f4",
        a_format=3 if block and tmem else 0,
        arch="sm_107a",
        mma_k=k,
        half_accumulator=half,
        lut_b=True,
        lut_segment=segment,
        **({"lut_address_offset": 112} if block else {}),
        **options,
    )
    fp8_bits = np.array([0, 0x38, 0x40, 0x44, 0x48, 0x4A, 0x4C, 0x4E], np.uint8)
    a = (np.arange(4 * m * k).reshape(4, m, k) % 7 - 3).astype(np.float32)
    a_bits = fp8_bits[np.abs(a).astype(np.int32)] | ((a < 0).astype(np.uint8) << 7)
    if block and tmem:
        # Table 66: K64 MXF8F6F4 TMEM A uses packed 6-bit containers.
        codes = np.array([0, 8, 16, 20], np.uint8)[np.abs(a).astype(np.int32)]
        codes |= (a < 0).astype(np.uint8) << 5
        a_bits = np.zeros_like(codes)
        for bank in range(4):
            for row in range(m):
                packed = sum(int(value) << (6 * i) for i, value in enumerate(codes[bank, row]))
                a_bits[bank, row, :48] = np.frombuffer(packed.to_bytes(48, "little"), np.uint8)
    b_bits = np.zeros((n, 64), np.uint8)
    lookup = np.zeros((2, 128, 8), np.uint8)
    b = np.empty((n, k), np.float32)
    for cta in range(group):
        for lut_row in range(2):
            values = np.roll(np.arange(8, dtype=np.float32), cta + lut_row)
            values *= -1 if (cta + lut_row) % 2 else 1
            lookup[cta, lut_row] = fp8_bits[np.abs(values).astype(np.int32)] | (
                (values < 0).astype(np.uint8) << 7
            )
            for row in range(lut_row * 8, lut_row * 8 + 8):
                global_row = cta * 16 + row
                indices = (np.arange(128) * 3 + row + cta) % 8
                indices[64:] = (indices[64:] + 5) % 8
                # Independent little-endian packing, including fields crossing bytes.
                packed = sum(int(index) << (3 * i) for i, index in enumerate(indices))
                b_bits[global_row, :48] = np.frombuffer(packed.to_bytes(48, "little"), np.uint8)
                b[global_row] = values[indices[segment * 64 : (segment + 1) * 64]]
    seed = np.full((m, n), 0.5, np.float32)
    metadata = lookup.view(np.uint32).reshape(2, 128, 2)
    if block:
        metadata = np.zeros((2, 128, 4), np.uint32)
        metadata[:, :, 2:] = lookup.view(np.uint32).reshape(2, 128, 2)
        lane = np.arange(128, dtype=np.uint32)
        for cta in range(group):
            metadata[cta, :, 0] = (126 + lane % 2 + cta) << 8  # SFA_ID = 1.
            metadata[cta, :, 1] = (127 + lane // 16 % 2) << 16  # SFB_ID = 2.
        a *= np.exp2((np.arange(m) % 2 - 1 + np.arange(m) // 128)[:, None])
        # CTA2 uses replicated, joint-N B scales, unlike the local-N lookup tables.
        b *= np.exp2((np.arange(n) // 16 % 2)[:, None])

    def encode_result(value):
        return (
            value.astype(np.float16).view(np.uint16).astype(np.int32)
            if half
            else value.view(np.int32)
        )

    expected = a[0] @ b.T + seed
    if group == 2 and not block:
        expected[129] = seed[129]
    args = {
        "a": a_bits.view(np.uint16),
        "b": b_bits.view(np.uint16),
        "zero_mask": np.zeros(1, np.uint64),
        "metadata": metadata,
        "seed": encode_result(seed),
        "out": np.zeros((m, n), np.int32),
    }
    return kernel, args, encode_result(expected)


@pytest.mark.parametrize(
    "group,tmem,segment,half",
    [
        (1, False, 0, False),
        (1, True, 1, True),
        (2, False, 1, True),
        (2, True, 0, False),
    ],
)
def test_dense_lut_b_segments_and_cta_partition(group, tmem, segment, half, tmp_path):
    kernel, args, expected = lut_b_case(group, tmem, segment, half)
    for disabled in (True, False):
        args["zero_mask"][0] = np.uint64(1 << 63 if disabled else 0)
        outputs = run_checked(kernel, args, cache_dir=tmp_path).outputs
        np.testing.assert_array_equal(outputs["out"], args["seed"] if disabled else expected)

@pytest.mark.parametrize("resource", ["b", "lookup"])
def test_lut_b_async_read_lifetimes(resource):
    for block in (False, True):
        kernel, args, _ = lut_b_case(segment=1, block=block, early_reuse=resource)
        # The shared write is in the unused first 24B half; the TMEM write
        # overlaps the lookup. Neither is retired by the A-only wait.
        for disabled in (True, False):
            args["zero_mask"][0] = np.uint64(1 << 63 if disabled else 0)
            report = racecheck(kernel, args)
            assert report.verdict == ("clean" if disabled else "error"), report.to_dict()


@pytest.mark.parametrize(
    "group,tmem,segment", [(1, False, 0), (1, True, 1), (2, False, 1), (2, True, 0)]
)
def test_block_lut_b_scales_and_packed_tmem(group, tmem, segment, tmp_path):
    kernel, args, expected = lut_b_case(group, tmem, segment, block=True)
    for disabled in (True, False):
        args["zero_mask"][0] = np.uint64(1 << 63 if disabled else 0)
        outputs = run_checked(kernel, args, cache_dir=tmp_path).outputs
        np.testing.assert_array_equal(outputs["out"], args["seed"] if disabled else expected)
