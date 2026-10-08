from __future__ import annotations

import tvm.backend.cuda.tile_primitive  # noqa: F401
from tirx_harness.numsim.transpiler import native_frontend
from tvm.tirx.operator.tile_primitive import list_registered_schedules


def test_numsim_registers_all_cuda_tile_primitive_operations():
    schedules = list_registered_schedules()
    cuda_tile_calls = {name for name, targets in schedules.items() if "cuda" in targets}
    tile_ops = native_frontend.registry()["tile_ops"]

    assert tile_ops
    assert tile_ops == sorted(set(tile_ops))
    assert set(tile_ops) == cuda_tile_calls
