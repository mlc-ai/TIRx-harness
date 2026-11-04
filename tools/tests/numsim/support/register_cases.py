"""Shared launch shape for ordinary and predicated register arithmetic."""

import numpy as np

import tvm_tirx
from tvm_tirx.script import tirx as T


def masked_arithmetic_case(forms, inputs, *, preserved, operand_dtype=None):
    """Pair each spelling with its source names; numerical oracles stay with callers."""
    size = len(inputs[preserved])
    carrier = inputs[preserved].dtype.name
    operand = operand_dtype or carrier

    def read(expression):
        return f'T.reinterpret({expression}, ty="{operand}")' if operand != carrier else expression

    stored = f'T.reinterpret(dst[0], ty="{carrier}")' if operand != carrier else "dst[0]"
    lines = []
    for row, (spelling, sources) in enumerate(forms):
        args = ", ".join(read(f"{name}[i]") for name in sources)
        masked = ", ".join(read(f"{name}[source]") for name in sources)
        destination = "dst[0]" if operand_dtype else f"output[{row}, i]"
        lines.append(f'    T.ptx["{spelling}"]({destination}, {args})')
        if operand_dtype:
            lines.append(f"    output[{row}, i] = {stored}")
        lines += [
            f"    dst[0] = {read(f'{preserved}[i]')}",
            f'    T.ptx["{spelling}"](dst[0], {masked}, pred=lane % 2 == 0, preserve_dst=True)',
            f"    output[{row + len(forms)}, i] = {stored}",
        ]
    rows = 2 * len(forms)
    parameters = [
        f'{name}: T.Tensor(({size},), "{value.dtype.name}")' for name, value in inputs.items()
    ]
    parameters.append(f'output: T.Tensor(({rows}, {size}), "{carrier}")')
    kernel = tvm_tirx.script.from_source(
        f'''
@T.function
def kernel({", ".join(parameters)}):
    T.device_entry(launch=T.cuda.LaunchConfig(grid={size // 32}, block=32))
    block = T.cuda.block_idx("x")
    lane = T.cuda.lane_id()
    i = block * 32 + lane
    source = T.Select(lane % 2 == 0, i, {size})
    dst = T.alloc_local((1,), "{operand}")
{chr(10).join(lines)}
''',
        {"T": T},
    )
    inputs["output"] = np.zeros((rows, size), inputs[preserved].dtype)
    return kernel, inputs
