"""multimem.ld_reduce / st / red forms shared by NumSim and live multi-GPU runs.

Every case is one warp per rank. ``mc`` is the multicast address of the
symmetric buffer whose rank-local replica is ``data``; ``out`` is rank local.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import tvm
from tvm.script import tirx as T

LANES = 32

_CARRIER = {
    "u32": "uint32", "s32": "int32", "b32": "uint32", "u64": "uint64", "s64": "int64",
    "b64": "uint64", "f16": "uint16", "bf16": "uint16", "f16x2": "uint32",
    "bf16x2": "uint32", "f32": "float32", "f64": "float64",
}


@dataclass(frozen=True)
class MultimemCase:
    name: str
    kind: str
    form: str
    registers: int
    carrier: str
    make_values: Callable[[np.random.Generator, int], np.ndarray]

    @property
    def elements(self) -> int:
        return LANES * self.registers

    @property
    def outputs(self) -> tuple[str, ...]:
        return ("out",) if self.kind == "ld_reduce" else ("data",)

    @property
    def lane_dtype(self) -> np.dtype:
        """The dtype of one reduced element."""
        element = self.form.rsplit(".", 1)[1]
        if element in ("f16x2", "bf16x2", "f16", "bf16"):
            return np.dtype(np.uint16)
        return np.dtype(f"u{np.dtype(self.carrier).itemsize}")

    @property
    def tolerance_ulps(self) -> int:
        """The GB200 switch's f16/bf16 sum rounding is not RNE; NumSim rounds to
        nearest even, so its half sums agree to within one ulp."""
        half = self.lane_dtype == np.uint16
        return 1 if self.kind == "ld_reduce" and ".add." in self.form and half else 0

    def prim_func(self):
        chain = "T.ptx." + ".".join(
            {"global": "global_", "and": "and_", "or": "or_"}.get(part, part).replace("::", "__")
            for part in self.form.replace("multimem.", "multimem_", 1).split(".")
        )
        regs = self.registers
        if self.kind == "ld_reduce":
            operands = ", ".join(f"out[lane * {regs} + {i}]" for i in range(regs))
            body = f"{chain}({operands}, mc.ptr_to([lane * {regs}]))"
        else:
            operands = ", ".join(f"src[lane * {regs} + {i}]" for i in range(regs))
            body = f"{chain}(mc.ptr_to([lane * {regs}]), {operands})"
            if self.kind == "st":
                body = f"if rank == 0:\n        {body}"
        n, dtype = self.elements, self.carrier
        return tvm.script.from_source(
            f"""
@T.prim_func
def multimem_{self.name}(mc: T.Buffer(({n},), "{dtype}"), data: T.Buffer(({n},), "{dtype}"),
                         src: T.Buffer(({n},), "{dtype}"), out: T.Buffer(({n},), "{dtype}"),
                         rank: T.int32):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    {body}
""",
            {"T": T},
        )

    def rank_arguments(self, rank: int, world: int) -> dict[str, np.ndarray | np.int32]:
        rng = np.random.default_rng(1009 * rank + sum(map(ord, self.name)))
        values = self.make_values(rng, self.elements).astype(self.carrier)
        replica = values if self.kind == "ld_reduce" else self.make_values(
            np.random.default_rng(7 + sum(map(ord, self.name))), self.elements
        ).astype(self.carrier)
        return {
            "data": replica.copy(),
            "src": values.copy() if self.kind != "ld_reduce" else np.zeros_like(values),
            "out": np.zeros_like(values),
            "rank": np.int32(rank),
        }


def _f32_order_sensitive(rng, n):
    # Large cancelling magnitudes beside small terms expose the fold order.
    scale = rng.choice([1.0, 2.0**24, -(2.0**24), 3.0, 2.0**-149, -0.0], size=n)
    return (scale * rng.uniform(0.5, 2.0, size=n)).astype(np.float32)


def _f64_values(rng, n):
    scale = rng.choice([1.0, 2.0**53, -(2.0**53), 2.0**-1074, -0.0], size=n)
    return scale * rng.uniform(0.5, 2.0, size=n)


def _half_bits(fmt: str, packed: bool):
    def make(rng, n):
        count = n * (2 if packed else 1)
        if fmt == "f16":
            normals = rng.uniform(-8.0, 8.0, size=count).astype(np.float16).view(np.uint16)
            special = np.array([0x0000, 0x8000, 0x0001, 0x83FF, 0x7C00, 0xFC00, 0x7E00,
                                0x7BFF, 0xFBFF, 0x3C00, 0x0400], np.uint16)
        else:
            normals = (rng.uniform(-8.0, 8.0, size=count).astype(np.float32).view(np.uint32)
                       >> 16).astype(np.uint16)
            special = np.array([0x0000, 0x8000, 0x0001, 0x807F, 0x7F80, 0xFF80, 0x7FC0,
                                0x7F7F, 0xFF7F, 0x3F80, 0x0080], np.uint16)
        mask = rng.random(count) < 0.25
        bits = np.where(mask, rng.choice(special, size=count), normals).astype(np.uint16)
        return bits.view(np.uint32) if packed else bits

    return make


def _ints(dtype, low, high):
    def make(rng, n):
        values = rng.integers(low, high, size=n, dtype=np.int64, endpoint=True)
        return values.astype(dtype)

    return make


def _bits(dtype):
    def make(rng, n):
        return rng.integers(0, np.iinfo(dtype).max, size=n, dtype=dtype, endpoint=True)

    return make


# `red` sums must be exact under any arrival order: every fourth element holds
# only small subnormals (catching FTZ), the rest only small integers.
def _subnormal_elements(count):
    return np.arange(count) % 4 == 3


def _small_f32(rng, n):
    values = rng.integers(-64, 64, size=n).astype(np.float32)
    subnormal = _subnormal_elements(n)
    values[subnormal] = rng.choice([-1, 1, 2], size=int(subnormal.sum())) * np.float32(2.0**-140)
    return values


def _small_f64(rng, n):
    values = rng.integers(-64, 64, size=n).astype(np.float64)
    subnormal = _subnormal_elements(n)
    values[subnormal] = rng.choice([-1, 1, 2], size=int(subnormal.sum())) * 2.0**-1070
    return values


def _small_half(fmt: str):
    def make(rng, n):
        count = 2 * n
        exact = rng.integers(-16, 16, size=count).astype(np.float32)
        if fmt == "f16":
            bits = exact.astype(np.float16).view(np.uint16)
        else:
            bits = (exact.view(np.uint32) >> 16).astype(np.uint16)
        subnormal = _subnormal_elements(count)
        bits[subnormal] = rng.choice(np.array([0x0001, 0x0002, 0x8001], np.uint16),
                                     size=int(subnormal.sum()))
        return bits.view(np.uint32)

    return make


def _case(name, kind, form, registers, make_values):
    carrier = _CARRIER[form.rsplit(".", 1)[1]]
    return MultimemCase(name, kind, form, registers, carrier, make_values)


MULTIMEM_CASES = (
    _case("ldr_add_v4_f32", "ld_reduce", "multimem.ld_reduce.relaxed.sys.global.add.v4.f32",
          4, _f32_order_sensitive),
    _case("ldr_add_f32", "ld_reduce", "multimem.ld_reduce.weak.global.add.f32", 1,
          _f32_order_sensitive),
    _case("ldr_add_f64", "ld_reduce", "multimem.ld_reduce.global.add.f64", 1, _f64_values),
    _case("ldr_add_v4_f16x2", "ld_reduce", "multimem.ld_reduce.global.add.v4.f16x2", 4,
          _half_bits("f16", True)),
    _case("ldr_add_acc_v4_f16x2", "ld_reduce",
          "multimem.ld_reduce.global.add.acc::f32.v4.f16x2", 4, _half_bits("f16", True)),
    _case("ldr_add_v4_bf16x2", "ld_reduce", "multimem.ld_reduce.global.add.v4.bf16x2", 4,
          _half_bits("bf16", True)),
    _case("ldr_add_acc_v4_bf16x2", "ld_reduce",
          "multimem.ld_reduce.global.add.acc::f32.v4.bf16x2", 4, _half_bits("bf16", True)),
    _case("ldr_add_acc_v8_bf16", "ld_reduce",
          "multimem.ld_reduce.global.add.acc::f32.v8.bf16", 8, _half_bits("bf16", False)),
    _case("ldr_min_v4_f16x2", "ld_reduce", "multimem.ld_reduce.global.min.v4.f16x2", 4,
          _half_bits("f16", True)),
    _case("ldr_max_v4_bf16x2", "ld_reduce", "multimem.ld_reduce.global.max.v4.bf16x2", 4,
          _half_bits("bf16", True)),
    _case("ldr_add_u32", "ld_reduce", "multimem.ld_reduce.acquire.sys.global.add.u32", 1,
          _bits(np.uint32)),
    _case("ldr_min_s32", "ld_reduce", "multimem.ld_reduce.global.min.s32", 1,
          _ints(np.int32, -(2**31), 2**31 - 1)),
    _case("ldr_max_u64", "ld_reduce", "multimem.ld_reduce.global.max.u64", 1, _bits(np.uint64)),
    _case("ldr_min_s64", "ld_reduce", "multimem.ld_reduce.global.min.s64", 1,
          _ints(np.int64, -(2**62), 2**62)),
    _case("ldr_xor_b32", "ld_reduce", "multimem.ld_reduce.global.xor.b32", 1, _bits(np.uint32)),
    _case("ldr_and_b64", "ld_reduce", "multimem.ld_reduce.global.and.b64", 1, _bits(np.uint64)),
    _case("ldr_or_b64", "ld_reduce", "multimem.ld_reduce.global.or.b64", 1, _bits(np.uint64)),
    _case("st_v4_f32", "st", "multimem.st.relaxed.sys.global.v4.f32", 4, _f32_order_sensitive),
    _case("st_b64", "st", "multimem.st.global.b64", 1, _bits(np.uint64)),
    _case("st_f32", "st", "multimem.st.weak.global.f32", 1, _f32_order_sensitive),
    _case("red_add_v4_f32", "red", "multimem.red.relaxed.sys.global.add.v4.f32", 4, _small_f32),
    _case("red_add_f64", "red", "multimem.red.global.add.f64", 1, _small_f64),
    _case("red_add_v4_f16x2", "red", "multimem.red.global.add.v4.f16x2", 4, _small_half("f16")),
    _case("red_add_v4_bf16x2", "red", "multimem.red.release.sys.global.add.v4.bf16x2", 4,
          _small_half("bf16")),
    _case("red_add_u32", "red", "multimem.red.global.add.u32", 1, _bits(np.uint32)),
    _case("red_max_s32", "red", "multimem.red.global.max.s32", 1,
          _ints(np.int32, -(2**31), 2**31 - 1)),
    _case("red_min_u64", "red", "multimem.red.global.min.u64", 1, _bits(np.uint64)),
    _case("red_xor_b32", "red", "multimem.red.global.xor.b32", 1, _bits(np.uint32)),
)

CASES_BY_NAME = {case.name: case for case in MULTIMEM_CASES}
