"""Conversion masks/carriers must preserve independent, recorded GPU results."""

import numpy as np
import pytest
import tvm
from tvm.backend.cuda.ptx.table import TABLE, operand_dtypes
from tvm.script import tirx as T

from tests.numsim.microtests.cases.ptx_cvt_scalar import (
    SCALAR_CVT_COLUMNS,
    scalar_cvt_arguments,
)
from tests.numsim.microtests.cases.ptx_cvt_scalar_goldens import (
    SCALAR_CVT_GOLDENS,
    SCALAR_CVT_INPUTS,
)
from tests.numsim.runtime.test_ptx_cvt_packed_forms import _cvt_call
from tests.numsim.support.execution import run_checked
from tirx_harness.numsim.transpiler.ptx_dialect import decode_ptx_call


def scalar_predicate_case(name):
    forms = [
        (spelling, scalar_cvt_arguments(name)[buffer].dtype.name)
        for buffer, spellings in SCALAR_CVT_COLUMNS[name].items()
        for spelling in spellings
    ]
    source = np.full((len(forms), 32, 16), 0xAB, np.uint8)
    expected = np.full((3 * len(forms), 32, 16), 0x5A, np.uint8)
    lines = []
    for index, (spelling, canonical_destination) in enumerate(forms):
        destination_type, source_type = spelling.split(".")[-2:]
        original = scalar_cvt_arguments(name)[f"source_{source_type}"]
        call = decode_ptx_call(
            _cvt_call(spelling, canonical_destination, f'T.cast(0, "{original.dtype.name}")')[1]
        )
        entry = TABLE[call.op_name.removeprefix("tirx.ptx.")]
        # Table-owned legal carriers; exercise the widest integer source and
        # destination available, including a float payload carried in integer bits.
        carriers = [
            max(
                operand_dtypes(slot, call.modifiers),
                key=lambda dtype: tvm.DataType(dtype).bits,
            )
            for slot in entry.typed_operands
        ]
        destination_dtype, source_dtype = carriers
        dbytes, sbytes = (tvm.DataType(dtype).bits // 8 for dtype in carriers)
        result_bits = np.dtype(canonical_destination).itemsize * 8
        for lane in range(32):
            raw = SCALAR_CVT_INPUTS[source_type][lane // 2]
            source[index, lane, : original.itemsize] = list(
                raw.to_bytes(original.itemsize, "little")
            )
        lines += [
            f'    dst_{index} = T.alloc_local((1,), "{destination_dtype}")',
            f'    src_{index} = T.alloc_local((1,), "{source_dtype}")',
            f'    dst_bytes_{index} = dst_{index}.view("uint8")',
            f'    src_bytes_{index} = src_{index}.view("uint8")',
            "    if lane % 2 == 0:",
            f"        for byte in T.unroll({sbytes}):",
            f"            src_bytes_{index}[byte] = source[{index}, lane, byte]",
        ]
        for mode, predicate in enumerate(
            (
                "pred=lane % 2 == 0, preserve_dst=True",
                "pred=lane % 2 == 0",
                "pred=False, preserve_dst=True",
            )
        ):
            row = 3 * index + mode
            lines += [
                f"    for byte in T.unroll({dbytes}):",
                f"        dst_bytes_{index}[byte] = T.uint8(0x5A)",
                f'    T.ptx["{spelling}"](dst_{index}[0], src_{index}[0], {predicate})',
                f"    for byte in T.unroll({dbytes}):",
                f"        output[{row}, lane, byte] = dst_bytes_{index}[byte]",
            ]
            for lane in range(32):
                if lane % 2 == 0 and mode != 2:
                    value = SCALAR_CVT_GOLDENS[spelling][lane // 2]
                    if destination_type.startswith("s") and value & (1 << (result_bits - 1)):
                        value -= 1 << result_bits
                    value &= (1 << (8 * dbytes)) - 1
                    expected[row, lane, :dbytes] = list(value.to_bytes(dbytes, "little"))
                elif mode == 1:
                    expected[row, lane, :dbytes] = 0
    kernel = tvm.script.from_source(
        f"""@T.prim_func
def kernel(source: T.Buffer(({len(forms)}, 32, 16), "uint8"), output: T.Buffer(({len(expected)}, 32, 16), "uint8")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
{chr(10).join(lines)}
""",
        {"T": T},
    )
    return kernel, dict(source=source, output=np.full_like(expected, 0x5A)), expected


@pytest.mark.parametrize("name", tuple(SCALAR_CVT_COLUMNS))
def test_cvt_predicates_and_wide_carriers_match_gpu_goldens(name, tmp_path):
    kernel, inputs, expected = scalar_predicate_case(name)
    result = run_checked(kernel, inputs, cache_dir=tmp_path)
    np.testing.assert_array_equal(result.outputs["output"], expected)


def stochastic_predicate_case():
    from tests.numsim.microtests.cases.ptx_cvt_narrow_goldens import (
        F32_SOURCE,
        GOLDENS,
        RBITS_SOURCE,
    )

    lines = []
    expected = np.full((18, 256), 0x5A5A, np.uint32)
    for index, (destination, relu) in enumerate(
        (destination, relu)
        for destination in ("e2m1x4", "e4m3x4", "e5m2x4")
        for relu in (False, True)
    ):
        spelling = f"cvt.rs{'.relu' if relu else ''}.satfinite.{destination}.f32"
        golden = GOLDENS[f"{destination}.f32.rs{'.relu' if relu else ''}"]
        dtype = "uint16" if destination == "e2m1x4" else "uint32"
        lines.append(f'        dst_{index} = T.alloc_local((1,), "{dtype}")')
        for mode in range(3):
            row = 3 * index + mode
            offset = "256" if mode == 2 else "source_index"
            arguments = ", ".join(
                f"source[T.Select({offset} < 256, ({offset} + {part}) % 256, 256)]"
                for part in range(4)
            )
            predicate = "False" if mode == 2 else "lane % 2 == 0"
            preserve = ", preserve_dst=True" if mode != 1 else ""
            lines += [
                f"        dst_{index}[0] = T.{dtype}(0x5A5A)",
                f'        T.ptx["{spelling}"](dst_{index}[0], {arguments}, random[{offset}], '
                f"pred={predicate}{preserve})",
                f'        output[{row}, i] = T.cast(dst_{index}[0], "uint32")',
            ]
            if mode != 2:
                expected[row, ::2] = golden[::2]
            if mode == 1:
                expected[row, 1::2] = 0
    kernel = tvm.script.from_source(
        f"""@T.prim_func
def kernel(source: T.Buffer((256,), "float32"), random: T.Buffer((256,), "float32"), output: T.Buffer((18, 256), "uint32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    for step in range(8):
        i = step * 32 + lane
        source_index = T.Select(lane % 2 == 0, i, 256)
{chr(10).join(lines)}
""",
        {"T": T},
    )
    # rbits is deliberately carried in f32 storage: it is a bit payload, not
    # a numeric conversion, including random words that encode NaNs.
    inputs = dict(
        source=F32_SOURCE.view(np.float32),
        random=RBITS_SOURCE.view(np.float32),
        output=np.zeros_like(expected),
    )
    return kernel, inputs, expected


def test_stochastic_cvt_predicates_preserve_random_bits(tmp_path):
    kernel, inputs, expected = stochastic_predicate_case()
    result = run_checked(kernel, inputs, cache_dir=tmp_path)
    np.testing.assert_array_equal(result.outputs["output"], expected)
