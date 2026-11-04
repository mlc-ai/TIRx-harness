from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import tvm_tirx
from tirx_harness import numsim
from tirx_harness.numsim.dtype_abi import vector_dtype_abi
from tvm_tirx.script import tirx as T
from tvm_tirx.tirx.layout import Axis, S, TileLayout

_NUMPY_SCALAR_DTYPES = {
    "float16": np.float16,
    "float32": np.float32,
    "float64": np.float64,
    "int8": np.int8,
    "int16": np.int16,
    "int32": np.int32,
    "int64": np.int64,
    "uint8": np.uint8,
    "uint16": np.uint16,
    "uint32": np.uint32,
    "uint64": np.uint64,
}


def _make_complete_ordering_runtime_kernel():
    statements = [
        f"    T.ptx.fence.{semantics}.{scope}()"
        for semantics in ("sc", "acq_rel")
        for scope in ("cta", "cluster", "gpu", "sys")
    ]
    statements.extend(
        statement
        for increase in (False, True)
        for register_count in range(24, 257, 8)
        for statement in (
            "    T.ptx.setmaxnreg."
            f"{'inc' if increase else 'dec'}.sync.aligned.u32({register_count})",
            "    T.cuda.warpgroup_sync(7)",
        )
    )
    return tvm_tirx.script.from_source(
        "\n".join(
            (
                "@T.function",
                "def ordering_complete_runtime_domain(output: T.Tensor((1,), 'int32')):",
                "    T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=128))",
                "    warp = T.cuda.warp_id()",
                "    lane = T.cuda.lane_id()",
                *statements,
                "    if (warp == 0) and (lane == 0):",
                "        output[0] = 0x13579BDF",
            )
        ),
        extra_vars={"T": T},
    )


ORDERING_COMPLETE_RUNTIME_DOMAIN = _make_complete_ordering_runtime_kernel()


_SNAPSHOT_TRANSPOSED_LDSTMATRIX_LAYOUT = TileLayout(
    S[
        (4, 2, 4, 2, 2, 8) : (
            1 @ Axis.get("laneid"),
            1,
            1 @ Axis.get("wid_in_wg"),
            4,
            2,
            4 @ Axis.get("laneid"),
        )
    ]
)
_SNAPSHOT_BOUND_LDSTMATRIX_LAYOUT = TileLayout(
    S[
        (4, 2, 8, 1, 4, 2) : (
            1 @ Axis.get("wid_in_wg"),
            2,
            4 @ Axis.get("laneid"),
            4,
            1 @ Axis.get("laneid"),
            1,
        )
    ]
)


_RAW_TCGEN_LDST_LAYOUT = TileLayout(S[(128, 512) : (1 @ Axis.get("TLane"), 1 @ Axis.get("TCol"))])


_RAW_TCGEN_LDST_DOMAINS = {
    "16x32bx2": (1, 128),
    "16x64b": (1, 128),
    "16x128b": (2, 64),
    "16x256b": (4, 32),
    "32x32b": (1, 128),
}


def _make_raw_tcgen_ldst_runtime_kernel(shape: str, registers_per_num: int, num: int):
    statements = []
    registers = ", ".join(f"registers[{index}]" for index in range(registers_per_num * num))
    for packed in (False, True):
        ld_chain = f"tcgen05.ld.sync.aligned.{shape}.x{num}{'.pack::16b' if packed else ''}.b32"
        st_chain = f"tcgen05.st.sync.aligned.{shape}.x{num}{'.unpack::16b' if packed else ''}.b32"
        split_offset = 2 * num if packed else num
        ld_operands = f"{registers}, T.uint32(0)"
        st_operands = f"T.uint32(0), {registers}"
        if shape == "16x32bx2":
            ld_operands = f"{ld_operands}, {split_offset}"
            st_operands = f"T.uint32(0), {split_offset}, {registers}"
        statements.extend(
            (
                f'    T.ptx["{ld_chain}"]({ld_operands})',
                "    T.ptx.tcgen05.wait__ld.sync.aligned()",
                f'    T.ptx["{st_chain}"]({st_operands})',
                "    T.ptx.tcgen05.wait__st.sync.aligned()",
            )
        )
    name = shape.replace("x", "_x")
    return tvm_tirx.script.from_source(
        "\n".join(
            (
                "@T.function",
                f"def raw_tcgen_ldst_runtime_domain_{name}_{num}(output: T.Tensor((1,), 'int32')):",
                "    T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=128))",
                "    warp = T.cuda.warp_in_warpgroup()",
                "    lane = T.cuda.lane_id()",
                "    row = T.meta_var(warp * 32 + lane)",
                "    tmem = T.tmem((128, 512), 'uint32', allocated_addr=0, layout=TMEM_LAYOUT)",
                "    registers = T.alloc_local((128,), 'uint32')",
                "    for col in T.serial(512):",
                "        tmem[row, col] = T.cast(row * 257 + col, 'uint32')",
                "    T.cuda.cta_sync()",
                *statements,
                "    if (warp == 0) and (lane == 0):",
                "        output[0] = 0x2468ACE",
            )
        ),
        extra_vars={"T": T, "TMEM_LAYOUT": _RAW_TCGEN_LDST_LAYOUT},
    )


RAW_TCGEN_LDST_RUNTIME_DOMAINS = {
    (shape, num): _make_raw_tcgen_ldst_runtime_kernel(shape, registers_per_num, num)
    for shape, (registers_per_num, maximum_num) in _RAW_TCGEN_LDST_DOMAINS.items()
    for num in (1 << exponent for exponent in range(maximum_num.bit_length()))
}


_RAW_TCGEN_CP_SHAPE_ROUTES = (
    ("32x128b", "warpx4"),
    ("64x128b", "warpx2::02_13"),
    ("64x128b", "warpx2::01_23"),
    ("128x128b", ""),
    ("128x256b", ""),
    ("4x256b", ""),
)
_RAW_TCGEN_CP_DECOMPRESSIONS = (
    "",
    "b8x16.b4x16_p64",
    "b8x16.b6x16_p32",
)


def _raw_tcgen_cp_descriptor_fields(shape: str, cta_group: int) -> tuple[int, int, int]:
    if shape == "4x256b":
        return (8, 0, 0)
    if shape == "128x256b" or cta_group == 2:
        return (1, 64, 3)
    return (0, 8, 0)


def _make_raw_tcgen_cp_runtime_kernel(shape: str, multicast: str, cta_group: int, decompress: str):
    ldo, sdo, swizzle = _raw_tcgen_cp_descriptor_fields(shape, cta_group)
    cp_chain = f"tcgen05.cp.cta_group::{cta_group}.{shape}"
    if multicast:
        cp_chain = f"{cp_chain}.{multicast}"
    if decompress:
        cp_chain = f"{cp_chain}.{decompress}"
    slug = "_".join(
        item.replace("::", "_").replace(".", "_").replace("x", "x") or "none"
        for item in (shape, multicast, str(cta_group), decompress)
    )
    return tvm_tirx.script.from_source(
        "\n".join(
            (
                "@T.function",
                f"def raw_tcgen_cp_runtime_domain_{slug}(output: T.Tensor((1,), 'int32')):",
                "    T.device_entry(launch=T.cuda.LaunchConfig(grid=2, block=128, cluster=2))",
                "    cta = T.cuda.cluster_cta_id('x')",
                "    warp = T.cuda.warp_in_warpgroup()",
                "    lane = T.cuda.lane_id()",
                "    thread = T.meta_var(warp * 32 + lane)",
                "    shared = T.alloc_shared((16384,), 'uint8', align=128)",
                "    tmem = T.tmem((128, 512), 'uint32', allocated_addr=0, layout=TMEM_LAYOUT)",
                "    descriptor: T.uint64",
                "    for index in T.serial(128):",
                "        shared[thread + index * 128] = T.uint8(0)",
                "    T.cuda.cluster_sync()",
                "    if (cta == 0) and (warp == 0) and (lane == 0):",
                "        T.cuda.tcgen05.encode_matrix_descriptor(",
                f"            T.address_of(descriptor), T.address_of(shared[0]), {ldo}, {sdo}, {swizzle}",
                "        )",
                "        descriptor_value: T.let = descriptor",
                f'        T.ptx["{cp_chain}"](T.uint32(0), descriptor)',
                f'        T.ptx["{cp_chain}"](T.uint32(0), descriptor_value)',
                "    T.cuda.cluster_sync()",
                "    if (cta == 0) and (warp == 0) and (lane == 0):",
                "        output[0] = T.cast(tmem[0, 0] == T.uint32(0), 'int32')",
            )
        ),
        extra_vars={"T": T, "TMEM_LAYOUT": _RAW_TCGEN_LDST_LAYOUT},
    )


RAW_TCGEN_CP_RUNTIME_DOMAINS = {
    (shape, multicast, cta_group, decompress): _make_raw_tcgen_cp_runtime_kernel(
        shape, multicast, cta_group, decompress
    )
    for shape, multicast in _RAW_TCGEN_CP_SHAPE_ROUTES
    for cta_group in (1, 2)
    for decompress in _RAW_TCGEN_CP_DECOMPRESSIONS
}


def _packed_storage(itemsize: int, *, initialized: bool) -> np.ndarray:
    if initialized:
        raw = (np.arange(itemsize, dtype=np.uint16) * np.uint16(37) + np.uint16(11)).astype(
            np.uint8
        )
    else:
        raw = np.zeros(itemsize, dtype=np.uint8)
    if itemsize == 2:
        return raw.view(np.uint16)
    if itemsize == 4:
        return raw.view(np.uint32)
    if itemsize == 8:
        return raw.view(np.uint64)
    if itemsize == 16:
        return raw.view(np.dtype("V16"))
    raise AssertionError(f"unexpected accepted CUDA ldg itemsize {itemsize}")


def _storage(dtype: str, *, initialized: bool) -> tuple[np.ndarray, bool]:
    vector_abi = vector_dtype_abi(dtype)
    if vector_abi is not None:
        return _packed_storage(vector_abi.itemsize, initialized=initialized), True
    if dtype == "bfloat16":
        bits = 0x3FC0 if initialized else 0
        return np.array([bits], dtype=np.uint16), True
    numpy_dtype = _NUMPY_SCALAR_DTYPES[dtype]
    value = 1.5 if dtype.startswith("float") else 7
    return np.array([value if initialized else 0], dtype=numpy_dtype), False


def _assert_same_physical_value(actual: np.ndarray, expected: np.ndarray) -> None:
    if actual.dtype.kind == "V":
        np.testing.assert_array_equal(actual.view(np.uint8), expected.view(np.uint8))
    else:
        np.testing.assert_array_equal(actual, expected)


def _bfloat16_bits(values: np.ndarray) -> np.ndarray:
    raw = np.asarray(values, dtype=np.float32).view(np.uint32)
    rounded = raw + np.uint32(0x7FFF) + ((raw >> np.uint32(16)) & np.uint32(1))
    return (rounded >> np.uint32(16)).astype(np.uint16)


def _transposed_ldstmatrix_expected(source_registers: np.ndarray) -> np.ndarray:
    expected = np.empty((8, 128), dtype=np.uint16)
    factors = (4, 2, 4, 2, 2, 8)
    for row in range(8):
        for column in range(128):
            coordinates = np.unravel_index(row * 128 + column, factors)
            lane = coordinates[0] + 4 * coordinates[5]
            warp = coordinates[2]
            register = coordinates[1] + 4 * coordinates[3] + 2 * coordinates[4]
            expected[row, column] = source_registers[warp * 32 + lane, register]
    return expected


def _bound_ldstmatrix_expected(source_registers: np.ndarray) -> np.ndarray:
    tile = np.empty((64, 8), dtype=np.uint16)
    factors = (4, 2, 8, 1, 4, 2)
    for row in range(64):
        for column in range(8):
            coordinates = np.unravel_index(row * 8 + column, factors)
            warp = coordinates[0]
            lane = 4 * coordinates[2] + coordinates[4]
            register = 2 * coordinates[1] + 4 * coordinates[3] + coordinates[5]
            tile[row, column] = source_registers[warp * 32 + lane, register]
    return np.tile(tile, (1, 8))


def test_ordering_fence_and_register_policy_finite_domains_execute(tmp_path: Path):
    module = numsim.transpile(ORDERING_COMPLETE_RUNTIME_DOMAIN, cache_dir=tmp_path)
    result = numsim.Engine(max_workers=1).run(module, {"output": np.zeros(1, dtype=np.int32)})

    np.testing.assert_array_equal(result.outputs["output"], np.array([0x13579BDF], dtype=np.int32))


@pytest.mark.parametrize(
    ("shape", "num"),
    sorted(RAW_TCGEN_LDST_RUNTIME_DOMAINS),
    ids=lambda value: str(value),
)
def test_raw_tcgen_ldst_finite_domain_executes_over_initialized_tmem(
    shape: str, num: int, tmp_path: Path
):
    module = numsim.transpile(
        RAW_TCGEN_LDST_RUNTIME_DOMAINS[shape, num], cache_dir=tmp_path / f"{shape}-{num}"
    )
    result = numsim.Engine(max_workers=1).run(module, {"output": np.zeros(1, dtype=np.int32)})

    np.testing.assert_array_equal(result.outputs["output"], np.array([0x2468ACE], dtype=np.int32))


@pytest.mark.parametrize(
    ("shape", "multicast", "cta_group", "decompress"),
    sorted(RAW_TCGEN_CP_RUNTIME_DOMAINS),
    ids=lambda value: str(value) or "none",
)
def test_raw_tcgen_cp_finite_domain_initializes_the_destination(
    shape: str,
    multicast: str,
    cta_group: int,
    decompress: str,
    tmp_path: Path,
):
    key = (shape, multicast, cta_group, decompress)
    slug = "-".join(str(item).replace("::", "_") or "none" for item in key)
    module = numsim.transpile(RAW_TCGEN_CP_RUNTIME_DOMAINS[key], cache_dir=tmp_path / slug)
    result = numsim.Engine(max_workers=1).run(module, {"output": np.zeros(1, dtype=np.int32)})

    np.testing.assert_array_equal(result.outputs["output"], np.ones(1, dtype=np.int32))
