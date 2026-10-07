from __future__ import annotations

import numpy as np
import tvm
from tvm.script import tirx as T

from tirx_harness import numsim


_REGISTER_BUFFER_COUNT = 5_600


def _many_register_buffer_kernel(count: int):
    lines = [
        "@T.prim_func",
        'def main(output: T.Buffer((1,), "int32")):',
        "    T.device_entry()",
        "    _warp = T.warp_id([1])",
        "    lane = T.lane_id([32])",
    ]
    lines.extend(f'    value_{index} = T.alloc_local((1,), "int32")' for index in range(count))
    # Unused register declarations remain part of the physical memory plan.
    # Keep execution itself tiny so this test isolates host artifact setup.
    lines.extend(
        (
            "    value_0[0] = lane + 17",
            "    if lane == 0:",
            "        output[0] = value_0[0]",
        )
    )
    return tvm.script.from_source("\n".join(lines), {"T": T})


def test_large_register_buffer_table_is_heap_backed_and_executes(tmp_path):
    # The former fixed arrays made prepare_kernel_0_buffers reserve more than
    # the default 8 MiB Linux thread stack at this size and SIGSEGV in its
    # stack-probe prologue. Metadata is now static and runtime tables are Vecs.
    module = numsim.transpile(
        _many_register_buffer_kernel(_REGISTER_BUFFER_COUNT), cache_dir=tmp_path
    )

    assert "buffers: Vec<RuntimeBuffer>" in module.rust_source
    assert "let register_backings = [" not in module.rust_source
    assert "let register_buffers = [" not in module.rust_source
    assert (
        f"persistent_buffers.extend(register_buffers[0..{_REGISTER_BUFFER_COUNT}].iter().cloned());"
    ) in module.rust_source

    result = numsim.Engine(max_workers=1).run(module, {"output": np.zeros(1, dtype=np.int32)})

    np.testing.assert_array_equal(result.outputs["output"], np.array([17], dtype=np.int32))
