"""PTX Figure 32 is an independent oracle for the non-power-of-two swizzle."""

import numpy as np
import pytest
import tvm
from tvm.script import tirx as T

from tests.numsim.support.execution import run_checked
from tests.numsim.runtime.test_raw_tma_codegen import _tensor_map


def swizzle96_kernel(offset=0, replace=False):
    mutation = (
        'T.ptx["tensormap_replace.tile.swizzle_mode.global.b1024.b32"](descriptor.ptr_to([0]), 4)'
        if replace
        else "T.evaluate(0)"
    )
    return tvm.script.from_source(
        f"""
@T.prim_func
def kernel(descriptor: T.Buffer((128,), "uint8"), out: T.Buffer((192,), "uint32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((224,), "uint32", scope="shared", align=256)
    barrier = T.alloc_buffer((1,), "uint64", scope="shared")
    if lane == 0:
        {mutation}
        {"T.ptx.fence.proxy.tensormap__generic.release.gpu()" if replace else "T.evaluate(0)"}
        T.ptx.fence.proxy.tensormap__generic.acquire.gpu(descriptor.ptr_to([0]))
        T.ptx.mbarrier.init.shared.b64(barrier.ptr_to([0]), 1)
    T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.cta_sync()
    if lane == 0:
        T.ptx["cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes"](
            shared.ptr_to([{offset // 4}]), descriptor.ptr_to([0]), 0, 0, barrier.ptr_to([0]))
        T.ptx.mbarrier.arrive.expect_tx.shared.b64(barrier.ptr_to([0]), 768)
        T.cuda.mbarrier_wait(barrier.ptr_to([0]), 0)
    T.cuda.cta_sync()
    for index in T.serial(6):
        out[index * 32 + lane] = shared[{offset // 4} + index * 32 + lane]
""",
        {"T": T},
    )


def swizzle96_inputs(replace):
    source = np.repeat(np.arange(64, dtype=np.uint32), 4).reshape(8, 32)
    descriptor, _ = _tensor_map(
        source,
        global_shape=(32, 8),
        global_strides=(128,),
        box_shape=(24, 8),
        swizzle=None if replace else "96B",
    )
    return {"descriptor": descriptor, "out": np.zeros(192, np.uint32)}


@pytest.mark.parametrize("offset,replace", [(0, False), (128, True)])
def test_swizzle96_documented_layout(offset, replace, tmp_path):
    kernel = swizzle96_kernel(offset, replace)
    result = run_checked(kernel, swizzle96_inputs(replace), cache_dir=tmp_path)
    # Transcribed from NVIDIA PTX Figure 32, each entry is one 16-byte atom.
    atoms = np.array(
        [
            0,
            1,
            2,
            3,
            4,
            5,
            8,
            9,
            11,
            10,
            13,
            12,
            17,
            16,
            19,
            18,
            20,
            21,
            24,
            25,
            26,
            27,
            28,
            29,
            33,
            32,
            35,
            34,
            37,
            36,
            41,
            40,
            42,
            43,
            44,
            45,
            48,
            49,
            50,
            51,
            53,
            52,
            57,
            56,
            59,
            58,
            61,
            60,
        ],
        np.uint32,
    )
    if offset:
        atoms = atoms.reshape(-1, 2)[:, ::-1].reshape(-1)
    np.testing.assert_array_equal(result.outputs["out"], np.repeat(atoms, 4))
