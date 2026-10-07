from __future__ import annotations

import numpy as np
import pytest

from tirx_harness import numsim
from tvm.script import tirx as T


def _make_int8_mma(a_dtype: str, b_dtype: str, k: int, *, saturate: bool):
    a_count = k // 8
    b_count = k // 16
    a_type = {"int8": "s8", "uint8": "u8"}[a_dtype]
    b_type = {"int8": "s8", "uint8": "u8"}[b_dtype]
    satfinite = ".satfinite" if saturate else ""
    instruction = f"mma.sync.aligned.m16n8k{k}.row.col{satfinite}.s32.{a_type}.{b_type}.s32"

    @T.prim_func
    def kernel(
        a: T.Buffer((16, k), a_dtype),
        b: T.Buffer((k, 8), b_dtype),
        c: T.Buffer((16, 8), "int32"),
        output: T.Buffer((16, 8), "int32"),
    ):
        T.device_entry()
        _warp = T.warp_id([1])
        lane = T.lane_id([32])
        group = T.meta_var(lane // 4)
        thread = T.meta_var(lane % 4)
        a_regs = T.alloc_local((a_count * 4,), a_dtype)
        b_regs = T.alloc_local((b_count * 4,), b_dtype)
        c_regs = T.alloc_local((4,), "int32")
        d_regs = T.alloc_local((4,), "int32")

        for register in T.unroll(a_count):
            row = T.meta_var(group + (register % 2) * 8)
            col = T.meta_var(thread * 4 + (register // 2) * 16)
            for packed in T.unroll(4):
                a_regs[register * 4 + packed] = a[row, col + packed]
        for register in T.unroll(b_count):
            row = T.meta_var(thread * 4 + register * 16)
            for packed in T.unroll(4):
                b_regs[register * 4 + packed] = b[row + packed, group]
        c_regs[0] = c[group, thread * 2]
        c_regs[1] = c[group, thread * 2 + 1]
        c_regs[2] = c[group + 8, thread * 2]
        c_regs[3] = c[group + 8, thread * 2 + 1]
        a_words = a_regs.view("uint32")
        b_words = b_regs.view("uint32")
        c_words = c_regs.view("uint32")
        d_words = d_regs.view("uint32")

        T.ptx[instruction](
            *[d_words[index] for index in range(4)],
            *[a_words[index] for index in range(a_count)],
            *[b_words[index] for index in range(b_count)],
            *[c_words[index] for index in range(4)],
        )

        output[group, thread * 2] = d_regs[0]
        output[group, thread * 2 + 1] = d_regs[1]
        output[group + 8, thread * 2] = d_regs[2]
        output[group + 8, thread * 2 + 1] = d_regs[3]

    return kernel


ptx_mma_s8_u8_m16n8k32 = _make_int8_mma("int8", "uint8", 32, saturate=False)
ptx_mma_u8_s8_m16n8k16_sat = _make_int8_mma("uint8", "int8", 16, saturate=True)


@T.prim_func
def ptx_mma_s8_u8_m16n8k32_no_c(
    a: T.Buffer((16, 32), "int8"), b: T.Buffer((32, 8), "uint8"), output: T.Buffer((16, 8), "int32")
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    group = T.meta_var(lane // 4)
    thread = T.meta_var(lane % 4)
    a_regs = T.alloc_local((16,), "int8")
    b_regs = T.alloc_local((8,), "uint8")
    d_regs = T.alloc_local((4,), "int32")
    for register in T.unroll(4):
        row = T.meta_var(group + (register % 2) * 8)
        col = T.meta_var(thread * 4 + (register // 2) * 16)
        for packed in T.unroll(4):
            a_regs[register * 4 + packed] = a[row, col + packed]
    for register in T.unroll(2):
        row = T.meta_var(thread * 4 + register * 16)
        for packed in T.unroll(4):
            b_regs[register * 4 + packed] = b[row + packed, group]
    a_words = a_regs.view("uint32")
    b_words = b_regs.view("uint32")
    d_words = d_regs.view("uint32")
    T.ptx.mma.sync.aligned.m16n8k32.row.col.s32.s8.u8.s32(
        *[d_words[index] for index in range(4)],
        *[a_words[index] for index in range(4)],
        *[b_words[index] for index in range(2)],
        T.uint32(0),
        T.uint32(0),
        T.uint32(0),
        T.uint32(0),
    )
    output[group, thread * 2] = d_regs[0]
    output[group, thread * 2 + 1] = d_regs[1]
    output[group + 8, thread * 2] = d_regs[2]
    output[group + 8, thread * 2 + 1] = d_regs[3]


@T.prim_func
def ptx_mma_legacy_s8_u8_m16n8k32(
    a: T.Buffer((16, 32), "int8"),
    b: T.Buffer((32, 8), "uint8"),
    c: T.Buffer((16, 8), "int32"),
    output: T.Buffer((16, 8), "int32"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    group = T.meta_var(lane // 4)
    thread = T.meta_var(lane % 4)
    a_regs = T.alloc_local((16,), "int8")
    b_regs = T.alloc_local((8,), "uint8")
    accumulator = T.alloc_local((4,), "int32")
    for register in T.unroll(4):
        row = T.meta_var(group + (register % 2) * 8)
        col = T.meta_var(thread * 4 + (register // 2) * 16)
        for packed in T.unroll(4):
            a_regs[register * 4 + packed] = a[row, col + packed]
    for register in T.unroll(2):
        row = T.meta_var(thread * 4 + register * 16)
        for packed in T.unroll(4):
            b_regs[register * 4 + packed] = b[row + packed, group]
    accumulator[0] = c[group, thread * 2]
    accumulator[1] = c[group, thread * 2 + 1]
    accumulator[2] = c[group + 8, thread * 2]
    accumulator[3] = c[group + 8, thread * 2 + 1]
    T.ptx_legacy.mma(
        "m16n8k32",
        "row",
        "col",
        "int8",
        "uint8",
        "int32",
        a_regs.data,
        0,
        b_regs.data,
        0,
        accumulator.data,
        0,
        False,
        dtype="int32",
    )
    output[group, thread * 2] = accumulator[0]
    output[group, thread * 2 + 1] = accumulator[1]
    output[group + 8, thread * 2] = accumulator[2]
    output[group + 8, thread * 2 + 1] = accumulator[3]


@T.prim_func
def ptx_mma_f16_accumulator_m16n8k8(
    a: T.Buffer((16, 8), "float16"),
    b: T.Buffer((8, 8), "float16"),
    c: T.Buffer((16, 8), "float16"),
    output: T.Buffer((16, 8), "float16"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    group = T.meta_var(lane // 4)
    thread = T.meta_var(lane % 4)
    a_regs = T.alloc_local((4,), "float16")
    b_regs = T.alloc_local((2,), "float16")
    c_regs = T.alloc_local((4,), "float16")
    d_regs = T.alloc_local((4,), "float16")
    for index in T.unroll(2):
        a_regs[index] = a[group, thread * 2 + index]
        a_regs[index + 2] = a[group + 8, thread * 2 + index]
        b_regs[index] = b[thread * 2 + index, group]
    c_regs[0] = c[group, thread * 2]
    c_regs[1] = c[group, thread * 2 + 1]
    c_regs[2] = c[group + 8, thread * 2]
    c_regs[3] = c[group + 8, thread * 2 + 1]
    a_words = a_regs.view("uint32")
    b_words = b_regs.view("uint32")
    c_words = c_regs.view("uint32")
    d_words = d_regs.view("uint32")
    T.ptx.mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16(
        d_words[0],
        d_words[1],
        a_words[0],
        a_words[1],
        b_words[0],
        c_words[0],
        c_words[1],
    )
    output[group, thread * 2] = d_regs[0]
    output[group, thread * 2 + 1] = d_regs[1]
    output[group + 8, thread * 2] = d_regs[2]
    output[group + 8, thread * 2 + 1] = d_regs[3]


@T.prim_func
def ptx_mma_f16_b32_backing_m16n8k8(
    a_words: T.Buffer((32, 2), "uint32"),
    b_words: T.Buffer((32, 1), "uint32"),
    c: T.Buffer((16, 8), "float32"),
    output: T.Buffer((16, 8), "float32"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    group = T.meta_var(lane // 4)
    thread = T.meta_var(lane % 4)
    a_regs = T.alloc_local((2,), "uint32")
    b_regs = T.alloc_local((1,), "uint32")
    c_regs = T.alloc_local((4,), "float32")
    d_regs = T.alloc_local((4,), "float32")
    for register in T.unroll(2):
        a_regs[register] = a_words[lane, register]
    b_regs[0] = b_words[lane, 0]
    c_regs[0] = c[group, thread * 2]
    c_regs[1] = c[group, thread * 2 + 1]
    c_regs[2] = c[group + 8, thread * 2]
    c_regs[3] = c[group + 8, thread * 2 + 1]
    T.ptx.mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32(
        *[d_regs[index] for index in range(4)],
        *[a_regs[index] for index in range(2)],
        b_regs[0],
        *[c_regs[index] for index in range(4)],
    )
    output[group, thread * 2] = d_regs[0]
    output[group, thread * 2 + 1] = d_regs[1]
    output[group + 8, thread * 2] = d_regs[2]
    output[group + 8, thread * 2 + 1] = d_regs[3]


@T.prim_func
def ptx_mma_tf32_m16n8k4(
    a: T.Buffer((16, 4), "float32"),
    b: T.Buffer((4, 8), "float32"),
    c: T.Buffer((16, 8), "float32"),
    output: T.Buffer((16, 8), "float32"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    group = T.meta_var(lane // 4)
    thread = T.meta_var(lane % 4)
    a_regs = T.alloc_local((2,), "float32")
    b_regs = T.alloc_local((1,), "float32")
    c_regs = T.alloc_local((4,), "float32")
    d_regs = T.alloc_local((4,), "float32")

    a_regs[0] = a[group, thread]
    a_regs[1] = a[group + 8, thread]
    b_regs[0] = b[thread, group]
    c_regs[0] = c[group, thread * 2]
    c_regs[1] = c[group, thread * 2 + 1]
    c_regs[2] = c[group + 8, thread * 2]
    c_regs[3] = c[group + 8, thread * 2 + 1]
    a_words = a_regs.view("uint32")
    b_words = b_regs.view("uint32")
    T.ptx.mma.sync.aligned.m16n8k4.row.col.f32.tf32.tf32.f32(
        *[d_regs[index] for index in range(4)],
        *[a_words[index] for index in range(2)],
        b_words[0],
        *[c_regs[index] for index in range(4)],
    )
    output[group, thread * 2] = d_regs[0]
    output[group, thread * 2 + 1] = d_regs[1]
    output[group + 8, thread * 2] = d_regs[2]
    output[group + 8, thread * 2 + 1] = d_regs[3]


@T.prim_func
def ptx_mma_f64_m8n8k4(
    a: T.Buffer((8, 4), "float64"),
    b: T.Buffer((4, 8), "float64"),
    c: T.Buffer((8, 8), "float64"),
    output: T.Buffer((8, 8), "float64"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    group = T.meta_var(lane // 4)
    thread = T.meta_var(lane % 4)
    a_reg = T.alloc_local((1,), "float64")
    b_reg = T.alloc_local((1,), "float64")
    c_regs = T.alloc_local((2,), "float64")
    d_regs = T.alloc_local((2,), "float64")
    a_reg[0] = a[group, thread]
    b_reg[0] = b[thread, group]
    c_regs[0] = c[group, thread * 2]
    c_regs[1] = c[group, thread * 2 + 1]
    T.ptx.mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64(
        d_regs[0], d_regs[1], a_reg[0], b_reg[0], c_regs[0], c_regs[1]
    )
    output[group, thread * 2] = d_regs[0]
    output[group, thread * 2 + 1] = d_regs[1]


@T.prim_func
def ptx_mma_f64_m16n8k16(
    a: T.Buffer((16, 16), "float64"),
    b: T.Buffer((16, 8), "float64"),
    c: T.Buffer((16, 8), "float64"),
    output: T.Buffer((16, 8), "float64"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    group = T.meta_var(lane // 4)
    thread = T.meta_var(lane % 4)
    a_regs = T.alloc_local((8,), "float64")
    b_regs = T.alloc_local((4,), "float64")
    c_regs = T.alloc_local((4,), "float64")
    d_regs = T.alloc_local((4,), "float64")
    for register in T.unroll(8):
        a_regs[register] = a[group + 8 * (register % 2), 4 * (register // 2) + thread]
    for register in T.unroll(4):
        b_regs[register] = b[4 * register + thread, group]
    c_regs[0] = c[group, thread * 2]
    c_regs[1] = c[group, thread * 2 + 1]
    c_regs[2] = c[group + 8, thread * 2]
    c_regs[3] = c[group + 8, thread * 2 + 1]
    T.ptx.mma.sync.aligned.m16n8k16.row.col.f64.f64.f64.f64(
        *[d_regs[index] for index in range(4)],
        *[a_regs[index] for index in range(8)],
        *[b_regs[index] for index in range(4)],
        *[c_regs[index] for index in range(4)],
    )
    output[group, thread * 2] = d_regs[0]
    output[group, thread * 2 + 1] = d_regs[1]
    output[group + 8, thread * 2] = d_regs[2]
    output[group + 8, thread * 2 + 1] = d_regs[3]


def _make_packed_integer_word_mma(
    a_type: str, b_type: str, k: int, *, bit_op: str | None, saturate: bool = False
):
    bits = {"int4": 4, "uint4": 4, "int1": 1}[a_type]
    a_count = 16 * k * bits // (32 * 32)
    b_count = k * 8 * bits // (32 * 32)
    a_ptx_type = {"int4": "s4", "uint4": "u4", "int1": "b1"}[a_type]
    b_ptx_type = {"int4": "s4", "uint4": "u4", "int1": "b1"}[b_type]
    satfinite = ".satfinite" if saturate else ""
    bit_tokens = f".{bit_op}.popc" if bit_op is not None else ""
    instruction = (
        f"mma.sync.aligned.m16n8k{k}.row.col{satfinite}.s32."
        f"{a_ptx_type}.{b_ptx_type}.s32{bit_tokens}"
    )

    @T.prim_func
    def kernel(
        a_words: T.Buffer((32, a_count), "uint32"),
        b_words: T.Buffer((32, b_count), "uint32"),
        c: T.Buffer((16, 8), "int32"),
        output: T.Buffer((16, 8), "int32"),
    ):
        T.device_entry()
        _warp = T.warp_id([1])
        lane = T.lane_id([32])
        group = T.meta_var(lane // 4)
        thread = T.meta_var(lane % 4)
        a_regs = T.alloc_local((a_count,), "uint32")
        b_regs = T.alloc_local((b_count,), "uint32")
        c_regs = T.alloc_local((4,), "int32")
        d_regs = T.alloc_local((4,), "int32")
        for register in T.unroll(a_count):
            a_regs[register] = a_words[lane, register]
        for register in T.unroll(b_count):
            b_regs[register] = b_words[lane, register]
        c_regs[0] = c[group, thread * 2]
        c_regs[1] = c[group, thread * 2 + 1]
        c_regs[2] = c[group + 8, thread * 2]
        c_regs[3] = c[group + 8, thread * 2 + 1]
        c_words = c_regs.view("uint32")
        d_words = d_regs.view("uint32")
        T.ptx[instruction](
            *[d_words[index] for index in range(4)],
            *[a_regs[index] for index in range(a_count)],
            *[b_regs[index] for index in range(b_count)],
            *[c_words[index] for index in range(4)],
        )
        output[group, thread * 2] = d_regs[0]
        output[group, thread * 2 + 1] = d_regs[1]
        output[group + 8, thread * 2] = d_regs[2]
        output[group + 8, thread * 2 + 1] = d_regs[3]

    return kernel


ptx_mma_s4_u4_m16n8k32 = _make_packed_integer_word_mma("int4", "uint4", 32, bit_op=None)
ptx_mma_s4_u4_m16n8k32_sat = _make_packed_integer_word_mma(
    "int4", "uint4", 32, bit_op=None, saturate=True
)
ptx_mma_b1_xor_m16n8k128 = _make_packed_integer_word_mma("int1", "int1", 128, bit_op="xor")


@T.prim_func
def ptx_mma_e4m3_e5m2_m16n8k32(
    a_words: T.Buffer((32, 4), "uint32"),
    b_words: T.Buffer((32, 2), "uint32"),
    c: T.Buffer((16, 8), "float32"),
    output: T.Buffer((16, 8), "float32"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    group = T.meta_var(lane // 4)
    thread = T.meta_var(lane % 4)
    a_regs = T.alloc_local((4,), "uint32")
    b_regs = T.alloc_local((2,), "uint32")
    c_regs = T.alloc_local((4,), "float32")
    d_regs = T.alloc_local((4,), "float32")
    for register in T.unroll(4):
        a_regs[register] = a_words[lane, register]
    for register in T.unroll(2):
        b_regs[register] = b_words[lane, register]
    c_regs[0] = c[group, thread * 2]
    c_regs[1] = c[group, thread * 2 + 1]
    c_regs[2] = c[group + 8, thread * 2]
    c_regs[3] = c[group + 8, thread * 2 + 1]
    T.ptx.mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e5m2.f32(
        *[d_regs[index] for index in range(4)],
        *[a_regs[index] for index in range(4)],
        *[b_regs[index] for index in range(2)],
        *[c_regs[index] for index in range(4)],
    )
    output[group, thread * 2] = d_regs[0]
    output[group, thread * 2 + 1] = d_regs[1]
    output[group + 8, thread * 2] = d_regs[2]
    output[group + 8, thread * 2 + 1] = d_regs[3]


@T.prim_func
def ptx_mma_e5m2_e4m3_m16n8k16_f16(
    a_words: T.Buffer((32, 2), "uint32"),
    b_words: T.Buffer((32, 1), "uint32"),
    c: T.Buffer((16, 8), "float16"),
    output: T.Buffer((16, 8), "float16"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    group = T.meta_var(lane // 4)
    thread = T.meta_var(lane % 4)
    a_regs = T.alloc_local((2,), "uint32")
    b_regs = T.alloc_local((1,), "uint32")
    c_regs = T.alloc_local((4,), "float16")
    d_regs = T.alloc_local((4,), "float16")
    for register in T.unroll(2):
        a_regs[register] = a_words[lane, register]
    b_regs[0] = b_words[lane, 0]
    c_regs[0] = c[group, thread * 2]
    c_regs[1] = c[group, thread * 2 + 1]
    c_regs[2] = c[group + 8, thread * 2]
    c_regs[3] = c[group + 8, thread * 2 + 1]
    c_words = c_regs.view("uint32")
    d_words = d_regs.view("uint32")
    T.ptx.mma.sync.aligned.m16n8k16.row.col.f16.e5m2.e4m3.f16(
        d_words[0],
        d_words[1],
        *[a_regs[index] for index in range(2)],
        b_regs[0],
        c_words[0],
        c_words[1],
    )
    output[group, thread * 2] = d_regs[0]
    output[group, thread * 2 + 1] = d_regs[1]
    output[group + 8, thread * 2] = d_regs[2]
    output[group + 8, thread * 2 + 1] = d_regs[3]


@T.prim_func
def ptx_mma_f16_m8n8k4_four_computations(
    a: T.Buffer((4, 8, 4), "float16"),
    b: T.Buffer((4, 4, 8), "float16"),
    c: T.Buffer((4, 8, 8), "float32"),
    output: T.Buffer((4, 8, 8), "float32"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    computation = T.meta_var((lane % 16) // 4)
    high = T.meta_var(lane // 16)
    thread = T.meta_var(lane % 4)
    a_regs = T.alloc_local((4,), "float16")
    b_regs = T.alloc_local((4,), "float16")
    c_regs = T.alloc_local((8,), "float32")
    d_regs = T.alloc_local((8,), "float32")
    for index in T.unroll(4):
        a_regs[index] = a[computation, high * 4 + thread, index]
        b_regs[index] = b[computation, index, high * 4 + thread]
    for index in T.unroll(8):
        row = T.meta_var((thread & 1) + (index & 2) + high * 4)
        col = T.meta_var((index & 4) + (thread & 2) + (index & 1))
        c_regs[index] = c[computation, row, col]
    a_words = a_regs.view("uint32")
    b_words = b_regs.view("uint32")
    T.ptx.mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32(
        *[d_regs[index] for index in range(8)],
        a_words[0],
        a_words[1],
        b_words[0],
        b_words[1],
        *[c_regs[index] for index in range(8)],
    )
    for index in T.unroll(8):
        row = T.meta_var((thread & 1) + (index & 2) + high * 4)
        col = T.meta_var((index & 4) + (thread & 2) + (index & 1))
        output[computation, row, col] = d_regs[index]


def _pack_m16_words(values: np.ndarray, bits: int, *, operand: str) -> np.ndarray:
    rows, cols = values.shape
    k = cols if operand == "a" else rows
    count = (16 * k if operand == "a" else k * 8) * bits // (32 * 32)
    packed = np.zeros((32, count), dtype=np.uint32)
    mask = (1 << bits) - 1
    for row in range(rows):
        for col in range(cols):
            inner = col if operand == "a" else row
            output_axis = row if operand == "a" else col
            per_register = 32 // bits
            k_group = 4 * per_register
            if operand == "a":
                lane = 4 * (output_axis % 8) + (inner % k_group) // per_register
                register = 2 * (inner // k_group) + output_axis // 8
            else:
                lane = 4 * output_axis + (inner % k_group) // per_register
                register = inner // k_group
            element = inner % per_register
            packed[lane, register] |= np.uint32((int(values[row, col]) & mask) << (bits * element))
    return packed


def _tf32(values: np.ndarray) -> np.ndarray:
    bits = np.asarray(values, dtype=np.float32).view(np.uint32)
    finite = (bits & np.uint32(0x7F800000)) != np.uint32(0x7F800000)
    rounded = bits + np.uint32(0x00000FFF) + ((bits >> np.uint32(13)) & np.uint32(1))
    rounded &= np.uint32(0xFFFFE000)
    return np.where(finite, rounded, bits).astype(np.uint32).view(np.float32)


def test_m16n8k32_mixed_int8_mma_matches_integer_reference(tmp_path):
    a = (np.arange(16 * 32, dtype=np.int16).reshape(16, 32) % 15 - 7).astype(np.int8)
    b = (np.arange(32 * 8, dtype=np.uint16).reshape(32, 8) % 11).astype(np.uint8)
    c = (np.arange(16 * 8, dtype=np.int64).reshape(16, 8) * 17 - 400).astype(np.int32)
    module = numsim.transpile(ptx_mma_s8_u8_m16n8k32, cache_dir=tmp_path)
    result = numsim.Engine().run(module, {"a": a, "b": b, "c": c, "output": np.zeros_like(c)})
    expected = (a.astype(np.int64) @ b.astype(np.int64) + c.astype(np.int64)).astype(np.int32)
    np.testing.assert_array_equal(result.outputs["output"], expected)


def test_m16n8k32_no_c_uses_zero_accumulator(tmp_path):
    a = (np.arange(16 * 32, dtype=np.int16).reshape(16, 32) % 15 - 7).astype(np.int8)
    b = (np.arange(32 * 8, dtype=np.uint16).reshape(32, 8) % 11).astype(np.uint8)
    module = numsim.transpile(ptx_mma_s8_u8_m16n8k32_no_c, cache_dir=tmp_path)
    result = numsim.Engine().run(
        module, {"a": a, "b": b, "output": np.zeros((16, 8), dtype=np.int32)}
    )
    expected = (a.astype(np.int64) @ b.astype(np.int64)).astype(np.int32)
    np.testing.assert_array_equal(result.outputs["output"], expected)


def test_legacy_m16n8k32_int8_reuses_dense_form_and_engine(tmp_path):
    a = (np.arange(16 * 32, dtype=np.int16).reshape(16, 32) % 15 - 7).astype(np.int8)
    b = (np.arange(32 * 8, dtype=np.uint16).reshape(32, 8) % 11).astype(np.uint8)
    c = (np.arange(16 * 8, dtype=np.int64).reshape(16, 8) * 17 - 400).astype(np.int32)
    module = numsim.transpile(ptx_mma_legacy_s8_u8_m16n8k32, cache_dir=tmp_path)
    result = numsim.Engine().run(module, {"a": a, "b": b, "c": c, "output": np.zeros_like(c)})
    expected = (a.astype(np.int64) @ b.astype(np.int64) + c.astype(np.int64)).astype(np.int32)
    np.testing.assert_array_equal(result.outputs["output"], expected)


def test_m16n8k16_integer_satfinite_clamps_after_accumulation(tmp_path):
    a = np.full((16, 16), 255, dtype=np.uint8)
    b = np.full((16, 8), 127, dtype=np.int8)
    c = np.full((16, 8), np.iinfo(np.int32).max - 3, dtype=np.int32)
    module = numsim.transpile(ptx_mma_u8_s8_m16n8k16_sat, cache_dir=tmp_path)
    result = numsim.Engine().run(module, {"a": a, "b": b, "c": c, "output": np.zeros_like(c)})
    np.testing.assert_array_equal(result.outputs["output"], np.full_like(c, np.iinfo(np.int32).max))


def test_m16n8k8_f16_accumulator_uses_packed_output_registers(tmp_path):
    a = (np.arange(16 * 8, dtype=np.float16).reshape(16, 8) % 5 - 2).astype(np.float16)
    b = (np.arange(8 * 8, dtype=np.float16).reshape(8, 8) % 5 - 2).astype(np.float16)
    c = (np.arange(16 * 8, dtype=np.float16).reshape(16, 8) % 7 - 3).astype(np.float16)
    module = numsim.transpile(ptx_mma_f16_accumulator_m16n8k8, cache_dir=tmp_path)
    result = numsim.Engine().run(module, {"a": a, "b": b, "c": c, "output": np.zeros_like(c)})
    expected = (a.astype(np.float32) @ b.astype(np.float32) + c.astype(np.float32)).astype(
        np.float16
    )
    np.testing.assert_array_equal(result.outputs["output"], expected)


def test_m16n8k8_f16_accepts_b32_backed_input_registers(tmp_path):
    a = (np.arange(16 * 8, dtype=np.float16).reshape(16, 8) % 5 - 2).astype(np.float16)
    b = (np.arange(8 * 8, dtype=np.float16).reshape(8, 8) % 5 - 2).astype(np.float16)
    c = np.arange(16 * 8, dtype=np.float32).reshape(16, 8) / 16
    module = numsim.transpile(ptx_mma_f16_b32_backing_m16n8k8, cache_dir=tmp_path)
    result = numsim.Engine().run(
        module,
        {
            "a_words": _pack_m16_words(a.view(np.uint16), 16, operand="a"),
            "b_words": _pack_m16_words(b.view(np.uint16), 16, operand="b"),
            "c": c,
            "output": np.zeros_like(c),
        },
    )
    expected = a.astype(np.float32) @ b.astype(np.float32) + c
    np.testing.assert_array_equal(result.outputs["output"], expected)


def test_m16n8k4_tf32_mma_matches_tf32_reference(tmp_path):
    a = np.arange(16 * 4, dtype=np.float32).reshape(16, 4) % 9 - 4
    b = np.arange(4 * 8, dtype=np.float32).reshape(4, 8) % 7 - 3
    c = np.arange(16 * 8, dtype=np.float32).reshape(16, 8) / 32
    module = numsim.transpile(ptx_mma_tf32_m16n8k4, cache_dir=tmp_path)
    result = numsim.Engine().run(module, {"a": a, "b": b, "c": c, "output": np.zeros_like(c)})
    expected = _tf32(a) @ _tf32(b) + c
    np.testing.assert_array_equal(result.outputs["output"], expected)


def test_m8n8k4_f64_mma_matches_fused_reference(tmp_path):
    a = (np.arange(8 * 4, dtype=np.float64).reshape(8, 4) % 9 - 4) / 8
    b = (np.arange(4 * 8, dtype=np.float64).reshape(4, 8) % 7 - 3) / 4
    c = np.arange(8 * 8, dtype=np.float64).reshape(8, 8) / 64
    module = numsim.transpile(ptx_mma_f64_m8n8k4, cache_dir=tmp_path)
    result = numsim.Engine().run(module, {"a": a, "b": b, "c": c, "output": np.zeros_like(c)})
    expected = np.empty_like(c)
    for row in range(8):
        for col in range(8):
            value = c[row, col]
            for inner in range(4):
                value = np.float64(a[row, inner] * b[inner, col] + value)
            expected[row, col] = value
    np.testing.assert_array_equal(result.outputs["output"], expected)


def test_m16n8k16_f64_mma_matches_fused_reference(tmp_path):
    a = (np.arange(16 * 16, dtype=np.float64).reshape(16, 16) % 9 - 4) / 8
    b = (np.arange(16 * 8, dtype=np.float64).reshape(16, 8) % 7 - 3) / 4
    c = np.arange(16 * 8, dtype=np.float64).reshape(16, 8) / 64
    module = numsim.transpile(ptx_mma_f64_m16n8k16, cache_dir=tmp_path)
    result = numsim.Engine().run(module, {"a": a, "b": b, "c": c, "output": np.zeros_like(c)})
    expected = np.empty_like(c)
    for row in range(16):
        for col in range(8):
            value = c[row, col]
            for inner in range(16):
                value = np.float64(a[row, inner] * b[inner, col] + value)
            expected[row, col] = value
    np.testing.assert_array_equal(result.outputs["output"], expected)


def test_m16n8k32_int4_uses_packed_b32_fragment_abi(tmp_path):
    a = (np.arange(16 * 32, dtype=np.int16).reshape(16, 32) % 16 - 8).astype(np.int8)
    b = (np.arange(32 * 8, dtype=np.uint16).reshape(32, 8) % 16).astype(np.uint8)
    c = (np.arange(16 * 8, dtype=np.int64).reshape(16, 8) - 50).astype(np.int32)
    module = numsim.transpile(ptx_mma_s4_u4_m16n8k32, cache_dir=tmp_path)
    result = numsim.Engine().run(
        module,
        {
            "a_words": _pack_m16_words(a, 4, operand="a"),
            "b_words": _pack_m16_words(b, 4, operand="b"),
            "c": c,
            "output": np.zeros_like(c),
        },
    )
    expected = (a.astype(np.int64) @ b.astype(np.int64) + c.astype(np.int64)).astype(np.int32)
    np.testing.assert_array_equal(result.outputs["output"], expected)


def test_m16n8k32_int4_satfinite_clamps_after_accumulation(tmp_path):
    a = np.full((16, 32), 7, dtype=np.int8)
    b = np.full((32, 8), 15, dtype=np.uint8)
    c = np.full((16, 8), np.iinfo(np.int32).max - 1, dtype=np.int32)
    module = numsim.transpile(ptx_mma_s4_u4_m16n8k32_sat, cache_dir=tmp_path)
    result = numsim.Engine().run(
        module,
        {
            "a_words": _pack_m16_words(a, 4, operand="a"),
            "b_words": _pack_m16_words(b, 4, operand="b"),
            "c": c,
            "output": np.zeros_like(c),
        },
    )
    np.testing.assert_array_equal(result.outputs["output"], np.full_like(c, np.iinfo(np.int32).max))


def test_m16n8k128_b1_xor_popc_matches_logical_reference(tmp_path):
    a = ((np.arange(16 * 128).reshape(16, 128) * 3 + 1) % 5 < 2).astype(np.uint8)
    b = ((np.arange(128 * 8).reshape(128, 8) * 5 + 2) % 7 < 3).astype(np.uint8)
    c = (np.arange(16 * 8).reshape(16, 8) - 40).astype(np.int32)
    module = numsim.transpile(ptx_mma_b1_xor_m16n8k128, cache_dir=tmp_path)
    result = numsim.Engine().run(
        module,
        {
            "a_words": _pack_m16_words(a, 1, operand="a"),
            "b_words": _pack_m16_words(b, 1, operand="b"),
            "c": c,
            "output": np.zeros_like(c),
        },
    )
    expected = np.logical_xor(a[:, :, None], b[None, :, :]).sum(axis=1, dtype=np.int32) + c
    np.testing.assert_array_equal(result.outputs["output"], expected)


def test_m16n8k32_mixed_fp8_uses_packed_fragment_abi(tmp_path):
    ml_dtypes = pytest.importorskip("ml_dtypes")
    a = (np.arange(16 * 32, dtype=np.float32).reshape(16, 32) % 5 - 2).astype(
        ml_dtypes.float8_e4m3fn
    )
    b = (np.arange(32 * 8, dtype=np.float32).reshape(32, 8) % 5 - 2).astype(ml_dtypes.float8_e5m2)
    c = np.arange(16 * 8, dtype=np.float32).reshape(16, 8) / 16
    module = numsim.transpile(ptx_mma_e4m3_e5m2_m16n8k32, cache_dir=tmp_path)
    result = numsim.Engine().run(
        module,
        {
            "a_words": _pack_m16_words(a.view(np.uint8), 8, operand="a"),
            "b_words": _pack_m16_words(b.view(np.uint8), 8, operand="b"),
            "c": c,
            "output": np.zeros_like(c),
        },
    )
    expected = a.astype(np.float32) @ b.astype(np.float32) + c
    np.testing.assert_array_equal(result.outputs["output"], expected)


def test_m16n8k16_mixed_fp8_supports_f16_accumulator(tmp_path):
    ml_dtypes = pytest.importorskip("ml_dtypes")
    a = (np.arange(16 * 16, dtype=np.float32).reshape(16, 16) % 5 - 2).astype(ml_dtypes.float8_e5m2)
    b = (np.arange(16 * 8, dtype=np.float32).reshape(16, 8) % 5 - 2).astype(ml_dtypes.float8_e4m3fn)
    c = (np.arange(16 * 8, dtype=np.float16).reshape(16, 8) % 7 - 3).astype(np.float16)
    module = numsim.transpile(ptx_mma_e5m2_e4m3_m16n8k16_f16, cache_dir=tmp_path)
    result = numsim.Engine().run(
        module,
        {
            "a_words": _pack_m16_words(a.view(np.uint8), 8, operand="a"),
            "b_words": _pack_m16_words(b.view(np.uint8), 8, operand="b"),
            "c": c,
            "output": np.zeros_like(c),
        },
    )
    expected = (a.astype(np.float32) @ b.astype(np.float32) + c.astype(np.float32)).astype(
        np.float16
    )
    np.testing.assert_array_equal(result.outputs["output"], expected)


def test_m8n8k4_f16_executes_four_independent_mmas(tmp_path):
    a = (np.arange(4 * 8 * 4, dtype=np.float16).reshape(4, 8, 4) % 5 - 2).astype(np.float16)
    b = (np.arange(4 * 4 * 8, dtype=np.float16).reshape(4, 4, 8) % 5 - 2).astype(np.float16)
    c = np.arange(4 * 8 * 8, dtype=np.float32).reshape(4, 8, 8) / 16
    module = numsim.transpile(ptx_mma_f16_m8n8k4_four_computations, cache_dir=tmp_path)
    result = numsim.Engine().run(module, {"a": a, "b": b, "c": c, "output": np.zeros_like(c)})
    expected = (
        np.stack([a[index].astype(np.float32) @ b[index].astype(np.float32) for index in range(4)])
        + c
    )
    np.testing.assert_array_equal(result.outputs["output"], expected)
