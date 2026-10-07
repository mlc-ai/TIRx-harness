"""Fused TMEM compression shares the ordinary load's footprint and wait contract."""

import numpy as np
import pytest
import tvm
from tvm.script import tirx as T

from tirx_harness import racecheck
from tests.numsim.support.execution import run_checked


def compression_kernel(num=4, *, reduce=False, maximum=True, absolute=False, nan=False, wait=True):
    metadata = (num + 31) // 32
    count = metadata + num // 2 + int(reduce)
    modifiers = (".abs" if absolute else "") + (".NaN" if nan else "")
    operation = "max" if maximum else "min"
    name = f"tcgen05.ld{'.red' if reduce else ''}.spcompress.sync.aligned.32x32b.x{num}.{operation}.sp::2:4{modifiers}.f32.b2"
    inputs = ", ".join(f"r[{i}]" for i in range(num))
    outputs = ", ".join(f"result[{i}]" for i in range(count - int(reduce)))
    if reduce:
        outputs += ", reduced[0]"
    kernel = tvm.script.from_source(
        f'''
@T.prim_func
def kernel(source: T.Buffer((32, {num}), "float32"), output: T.Buffer((32, {count}), "uint32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    address = T.alloc_buffer((1,), "uint32", scope="shared")
    r = T.alloc_local(({num},), "float32")
    result = T.alloc_local(({count},), "uint32")
    reduced = T.alloc_local((1,), "float32")
    T.ptx.tcgen05.alloc.cta_group__1.sync.aligned.shared__cta.b32(T.address_of(address[0]), {max(32, num)})
    for i in T.unroll({num}):
        r[i] = source[lane, i]
    T.ptx["tcgen05.st.sync.aligned.32x32b.x{num}.b32"](address[0], {inputs})
    if {wait}:
        T.ptx.tcgen05.wait__st.sync.aligned()
    T.cuda.cta_sync()
    T.ptx["{name}"]({outputs}, address[0])
    T.ptx.tcgen05.wait__ld.sync.aligned()
    for i in T.unroll({count - int(reduce)}):
        output[lane, i] = result[i]
    if {reduce}:
        output[lane, {count - 1}] = T.reinterpret("uint32", reduced[0])
    T.ptx.tcgen05.dealloc.cta_group__1.sync.aligned.b32(address[0], {max(32, num)})
    T.ptx.tcgen05.relinquish_alloc_permit.cta_group__1.sync.aligned()
''',
        {"T": T},
    )
    return kernel.with_attr("tirx.cuda_arch", "sm_107a")


@pytest.mark.parametrize(
    "num,reduce,maximum,absolute,nan",
    [
        (4, False, True, False, False),
        (8, True, False, True, True),
        (128, True, True, False, False),
    ],
)
def test_tcgen_load_compression(num, reduce, maximum, absolute, nan, tmp_path):
    source = np.tile(np.array([-1, 4, -2, 3], np.float32), (32, num // 4))
    source[1, :4] = [np.nan, 4, np.nan, 3]
    count = (num + 31) // 32 + num // 2 + int(reduce)
    kernel = compression_kernel(num, reduce=reduce, maximum=maximum, absolute=absolute, nan=nan)
    args = {"source": source, "output": np.zeros((32, count), np.uint32)}
    actual = run_checked(kernel, args, cache_dir=tmp_path).outputs["output"]
    expected = np.zeros_like(actual)
    for lane in range(32):
        for group in range(num // 4):
            values = source[lane, group * 4 : group * 4 + 4]
            key = np.abs(values) if absolute else values
            # Distinct finite candidates and exactly two NaNs avoid tie-dependent choices.
            indices = sorted(
                range(4), key=lambda i: (not np.isnan(key[i]), -key[i] if maximum else key[i])
            )[:2]
            for j, index in enumerate(sorted(indices)):
                element = group * 2 + j
                expected[lane, element // 16] |= np.uint32(index << ((element % 16) * 2))
                expected[lane, (num + 31) // 32 + element] = values.view(np.uint32)[index]
        if reduce:
            values = np.abs(source[lane]) if absolute else source[lane]
            operation = (
                (np.maximum if maximum else np.minimum)
                if nan
                else (np.fmax if maximum else np.fmin)
            )
            value = operation.reduce(values)
            expected[lane, -1] = 0x7FFFFFFF if np.isnan(value) else value.view(np.uint32)
    np.testing.assert_array_equal(actual, expected)


def test_tcgen_compression_preserves_store_wait():
    args = {"source": np.ones((32, 4), np.float32), "output": np.zeros((32, 3), np.uint32)}
    report = racecheck(compression_kernel(wait=False), args)
    assert any(f.status == "error" and f.details["access_pair"] == "write_read" for f in report.findings)
