from __future__ import annotations

import tvm_ffi

import tvm_tirx  # noqa: F401
from tirx_harness.numsim.transpiler.host_prelude import _TILE_OPERATIONS


def test_numsim_dispatches_every_cuda_tile_operation():
    names = {str(name) for name in tvm_ffi.get_global_func("ir.ListOpNames")()}
    cuda_tile_operations = {name for name in names if name.startswith("tirx.cuda.tile.")}

    assert cuda_tile_operations
    assert cuda_tile_operations == _TILE_OPERATIONS
