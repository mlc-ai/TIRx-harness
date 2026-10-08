"""Scalar CVT launch and mask plumbing; callers own carriers and numerical oracles."""

import numpy as np
import tvm
from tvm.script import tirx as T


def append_cvt_mode(lines, expected, row, spelling, dtype, operand, converted, sentinel):
    """Emit ordinary, preserved, unpreserved, or false-predicate CVT by row."""
    mode = row % 4
    expected[row] = converted if mode == 0 else sentinel
    if mode in (1, 2):
        expected[row, ::2] = converted[::2]
        if mode == 2:
            expected[row, 1::2] = 0
    predicate = (
        "",
        ", pred=lane % 2 == 0, preserve_dst=True",
        ", pred=lane % 2 == 0",
        ", pred=False, preserve_dst=True",
    )[mode]
    initial = f"T.{dtype}({sentinel})"
    result = f"dst_{row}[0]"
    if dtype.startswith("float"):
        bits = dtype.removeprefix("float")
        initial = f'T.reinterpret("{dtype}", T.uint{bits}({sentinel}))'
        result = f'T.reinterpret("uint{bits}", {result})'
    lines.append(f'    dst_{row} = T.alloc_local((1,), "{dtype}")')
    if mode:
        lines.append(f"    dst_{row}[0] = {initial}")
    lines += [
        f'    T.ptx["{spelling}"](dst_{row}[0], {operand}{predicate})',
        f'    output[{row}, lane] = T.cast({result}, "uint64")',
    ]


def scalar_cvt_kernel(source, expected, lines):
    """Keep the existing two-warp CTA shape and raw uint64 input/output buffers."""
    size = source.shape[-1]
    kernel = tvm.script.from_source(
        f"""@T.prim_func
def kernel(source: T.Buffer({source.shape}, "uint64"), output: T.Buffer({expected.shape}, "uint64")):
    T.device_entry()
    block = T.cta_id([{size // 64}])
    warp = T.warp_id([2])
    local_lane = T.lane_id([32])
    lane = block * 64 + warp * 32 + local_lane
{chr(10).join(lines)}
""",
        {"T": T},
    )
    return kernel, dict(source=source, output=np.zeros_like(expected)), expected
