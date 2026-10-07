"""MOV copies payloads; predicates and sinks must not change register transport."""

from itertools import product

import numpy as np
import tvm
from tvm.script import tirx as T

from tests.numsim.support.execution import assert_rejected, run_checked


@T.prim_func
def mov_aliased_registers(
    source: T.Buffer((32, 4), "uint16"),
    output: T.Buffer((3, 32, 4), "uint16"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    words = T.alloc_local((4,), "uint16")
    wide = words.view("uint64")
    for i in T.unroll(4):
        words[i] = source[lane, i]
    T.ptx.mov.b64(
        words[3],
        words[2],
        words[1],
        words[0],
        wide[0],
        pred=words[3] != T.uint16(0),
        preserve_dst=True,
    )
    for i in T.unroll(4):
        output[0, lane, i] = words[i]
    T.ptx.mov.b64(
        wide[0],
        words[3],
        words[2],
        words[1],
        words[0],
        pred=words[0] != T.uint16(0),
        preserve_dst=True,
    )
    for i in T.unroll(4):
        output[1, lane, i] = words[i]
    # A sink is not a write to the corresponding aliased source field.
    T.ptx.mov.b64(T.ptx.SINK, words[0], T.ptx.SINK, words[2], wide[0], pred=True)
    for i in T.unroll(4):
        output[2, lane, i] = words[i]


def mov_alias_inputs():
    source = np.arange(128, dtype=np.uint16).reshape(32, 4)
    source[::2, 0] = 0
    source[1::2, 3] = 0
    expected = np.stack([source.copy() for _ in range(3)])
    expected[0, ::2] = source[::2, ::-1]
    expected[1] = expected[0]
    selected = expected[0, :, 0] != 0
    expected[1, selected] = expected[0, selected, ::-1]
    expected[2] = expected[1]
    expected[2, :, 0] = expected[1, :, 1]
    expected[2, :, 2] = expected[1, :, 3]
    return dict(source=source, output=np.zeros_like(expected)), expected


def test_mov_aliased_sources_destinations_and_predicate(tmp_path):
    inputs, expected = mov_alias_inputs()
    result = run_checked(mov_aliased_registers, inputs, cache_dir=tmp_path)
    np.testing.assert_array_equal(result.outputs["output"], expected)


def mov_pointer_case(space, *, dereference_null=False):
    size = 2 if space == "local" else 64
    index, second = ("0", "1") if space == "local" else ("lane", "lane + 32")
    declaration = (
        ""
        if space == "global"
        else (
            f'storage = T.alloc_buffer(({size},), "uint32", scope="{space}")\n'
            f"    storage[{index}] = source[lane]\n"
            f"    storage[{second}] = source[lane + 32]"
        )
    )
    storage = "source" if space == "global" else "storage"
    lines = []
    expected = np.full((12, 32), 7, np.uint32)
    for number, (ptx_type, mode) in enumerate(product(("b64", "u64"), range(6))):
        result = f"result_{number}"
        lines += [f'    {result} = T.alloc_local((1,), "uint64")']
        if mode == 5:
            lines += [
                f"    {result}[0] = T.uint64(0)",
                "    if lane % 2 == 0:",
                f"        {result}[0] = a",
            ]
        else:
            lines += [f"    {result}[0] = a"]
        predicate = (
            ""
            if mode == 0
            else ", pred=False, preserve_dst=True"
            if mode == 3
            else f", pred={result}[0] != T.uint64(0), preserve_dst=True"
            if mode == 5
            else f", pred=lane % 2 == 0, preserve_dst={mode != 2}"
        )
        source_value = "a" if mode == 0 else "T.uint64(0)" if mode == 4 else "partial[0]"
        lines += [f'    T.ptx["mov.{ptx_type}"]({result}[0], {source_value}{predicate})']
        guard = "lane % 2 == 0" if mode == 2 else f"{result}[0] != T.uint64(0)"
        if dereference_null and mode == 4:
            guard = "True"
        lines += [
            f"    if {guard}:",
            f'        T.ptx.ld.b32(output[{number}, lane], T.reinterpret("handle", {result}[0]))',
        ]
        if mode in (0, 3):
            expected[number] = np.arange(32, dtype=np.uint32) + 100
        elif mode in (1, 2, 5):
            if mode == 1:
                expected[number] = np.arange(32, dtype=np.uint32) + 100
            expected[number, ::2] = np.arange(0, 32, 2, dtype=np.uint32) + 132
        else:
            expected[number, 1::2] = np.arange(1, 32, 2, dtype=np.uint32) + 100
    kernel = tvm.script.from_source(
        f"""@T.prim_func
def kernel(source: T.Buffer((64,), "uint32"), output: T.Buffer((12, 32), "uint32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    {declaration}
    a = T.reinterpret("uint64", {storage}.ptr_to([{index}]))
    partial = T.alloc_local((1,), "uint64")
    if lane % 2 == 0:
        partial[0] = T.reinterpret("uint64", {storage}.ptr_to([{second}]))
{chr(10).join(lines)}
""",
        {"T": T},
    )
    return (
        kernel,
        dict(source=np.arange(64, dtype=np.uint32) + 100, output=np.full_like(expected, 7)),
        expected,
    )


def test_mov_pointer_identity_masks_and_nulls(tmp_path):
    for space in ("global", "shared"):
        kernel, inputs, expected = mov_pointer_case(space)
        result = run_checked(kernel, inputs, cache_dir=tmp_path)
        np.testing.assert_array_equal(result.outputs["output"], expected)
        kernel, inputs, _ = mov_pointer_case(space, dereference_null=True)
        assert_rejected(kernel, inputs, "null", cache_dir=tmp_path)
