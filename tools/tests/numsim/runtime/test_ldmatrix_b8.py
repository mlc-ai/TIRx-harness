"""PTX ldmatrix layouts, checked against matrix-level rather than engine formulas."""

import numpy as np
import pytest
import tvm
from tvm.script import tirx as T

from tests.numsim.support.execution import run_checked
from tirx_harness import racecheck

LDMATRIX_B8_CASES = [
    (bits, rows, count)
    for bits in (8, 6, 4)
    for rows in ((16,) if bits == 8 else (8, 16))
    for count in ((1, 2) if rows == 16 else (1, 2, 4))
]


def ldmatrix_b8_kernel(
    bits, rows, count, *, offset=0, synchronize=True, divergent=False, signed=False
):
    registers = count * rows // 8
    fmt = "b8" if bits == 8 else f"b8x16.b{bits}x16_p{128 - 16 * bits}"
    if signed:
        fmt = "s8.s4"
    instruction = f"ldmatrix.sync.aligned.m{rows}n16.x{count}"
    instruction += ".trans" if rows == 16 else ""
    instruction += f".shared::cta.{fmt}"
    destinations = ", ".join(f"fragment[{i}]" for i in range(registers))
    return tvm.script.from_source(
        f"""
@T.prim_func
def load_matrix(source: T.Buffer((32, 32), "uint8"), output: T.Buffer(({registers}, 32), "uint32")):
    T.device_entry()
    warp = T.warp_id([{1 if synchronize else 2}])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((32, 32), "uint8", scope="shared", align=16)
    fragment = T.alloc_local(({registers},), "uint32")
    if warp == 0:
        for byte in T.serial({bits * 2}):
            shared[lane, byte] = source[lane, byte]
    {"T.cuda.warp_sync()" if synchronize else "T.evaluate(0)"}
    if warp == {0 if synchronize else 1} and {"lane < 16" if divergent else "True"}:
        T.ptx["{instruction}"]({destinations}, shared.ptr_to([(lane * 5 + 3) % 32, {offset}]))
        for register in T.serial({registers}):
            output[register, lane] = fragment[register]
""",
        {"T": T},
    )


def ldmatrix_b8_case(bits, rows, count, *, signed=False):
    # Distinct rows, columns, high bits and crossed byte boundaries. Leave the
    # shared padding uninitialized: it contributes no output or read footprint.
    values = np.random.default_rng(2046).integers(0, 1 << bits, (32, 16), dtype=np.uint16)
    source = np.full((32, 32), 0xED, dtype=np.uint8)
    for row, elements in enumerate(values):
        packed = sum(int(value) << (column * bits) for column, value in enumerate(elements))
        source[row, : bits * 2] = np.frombuffer(packed.to_bytes(bits * 2, "little"), dtype=np.uint8)
    matrices = values[(np.arange(rows * count) * 5 + 3) % 32].reshape(count, rows, 16)
    if signed:
        matrices = np.where(matrices >= 8, matrices.astype(np.int16) - 16, matrices).astype(
            np.uint8
        )
    if rows == 16:
        matrices = matrices.transpose(0, 2, 1)
    # PTX fragment figure: each 8-row band is distributed to 32 lanes;
    # each lane holds four consecutive output bytes, least significant first.
    chunks = matrices.reshape(count * rows // 8, 32, 4).astype(np.uint32)
    expected = np.sum(chunks << np.array([0, 8, 16, 24], dtype=np.uint32), axis=2, dtype=np.uint32)
    return {"source": source, "output": np.zeros_like(expected)}, expected


@pytest.mark.parametrize("bits,rows,count", LDMATRIX_B8_CASES)
def test_ldmatrix_b8_layouts(bits, rows, count, tmp_path):
    kernel = ldmatrix_b8_kernel(bits, rows, count)
    inputs, expected = ldmatrix_b8_case(bits, rows, count)
    result = run_checked(kernel, inputs, cache_dir=tmp_path)
    np.testing.assert_array_equal(result.outputs["output"], expected)


@pytest.mark.parametrize("count", [1, 2, 4])
def test_ldmatrix_s8_s4(count, tmp_path):
    kernel = ldmatrix_b8_kernel(4, 8, count, signed=True)
    inputs, expected = ldmatrix_b8_case(4, 8, count, signed=True)
    result = run_checked(kernel, inputs, cache_dir=tmp_path)
    np.testing.assert_array_equal(result.outputs["output"], expected)


@pytest.mark.parametrize("mutation", ["unaligned", "divergent", "unordered"])
def test_ldmatrix_b8_invalid_controls(mutation):
    kernel = ldmatrix_b8_kernel(
        6,
        16,
        2,
        offset=1 if mutation == "unaligned" else 0,
        synchronize=mutation != "unordered",
        divergent=mutation == "divergent",
    )
    inputs, _ = ldmatrix_b8_case(6, 16, 2)
    report = racecheck(kernel, inputs)
    assert report.verdict == "error", report.format()
    if mutation == "unordered":
        assert any(finding.details["access_pair"] == "write_read" for finding in report.findings), report.format()
    elif mutation == "unaligned":
        assert "row address must be 16-byte aligned" in report.format()
    else:
        assert any(finding.kind == "warp_collective_divergence" for finding in report.findings), (
            report.format()
        )
