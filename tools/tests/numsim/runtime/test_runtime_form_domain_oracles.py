from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import tvm
from tvm.script import tirx as T
from tvm.script.tirx import tile as Tx
from tvm.tirx.layout import S, TCol, TileLayout, TLane, laneid, wg_local_layout, wid_in_wg

from tirx_harness import numsim
from tirx_harness.numsim.dtype_abi import vector_dtype_abi
from tests.numsim.support.runtime_domains import CUDA_LDG_DTYPES


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


def _make_complete_cuda_ldg_runtime_kernel():
    parameters = []
    statements = []
    for dtype in CUDA_LDG_DTYPES:
        name = dtype.replace("_", "")
        parameters.extend(
            (
                f'    source_{name}: T.Buffer((1,), "{dtype}"),',
                f'    output_{name}: T.Buffer((1,), "{dtype}"),',
            )
        )
        statements.append(
            f'        output_{name}[0] = T.cuda.ldg(source_{name}.ptr_to([0]), "{dtype}")'
        )
    return tvm.script.from_source(
        "\n".join(
            (
                "@T.prim_func",
                "def cuda_ldg_complete_runtime_domain(",
                *parameters,
                "):",
                "    T.device_entry()",
                "    _warp = T.warp_id([1])",
                "    lane = T.lane_id([32])",
                "    if lane == 0:",
                *statements,
            )
        ),
        extra_vars={"T": T},
    )


CUDA_LDG_COMPLETE_RUNTIME_DOMAIN = _make_complete_cuda_ldg_runtime_kernel()


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
    return tvm.script.from_source(
        "\n".join(
            (
                "@T.prim_func",
                "def ordering_complete_runtime_domain(output: T.Buffer((1,), 'int32')):",
                "    T.device_entry()",
                "    warp = T.warp_id([4])",
                "    lane = T.lane_id([32])",
                *statements,
                "    if (warp == 0) and (lane == 0):",
                "        output[0] = 0x13579BDF",
            )
        ),
        extra_vars={"T": T},
    )


ORDERING_COMPLETE_RUNTIME_DOMAIN = _make_complete_ordering_runtime_kernel()


_SNAPSHOT_TRANSPOSED_LDSTMATRIX_LAYOUT = TileLayout(
    S[(4, 2, 4, 2, 2, 8) : (1 @ laneid, 1, 1 @ wid_in_wg, 4, 2, 4 @ laneid)]
)
_SNAPSHOT_BOUND_LDSTMATRIX_LAYOUT = TileLayout(
    S[(4, 2, 8, 1, 4, 2) : (1 @ wid_in_wg, 2, 4 @ laneid, 4, 1 @ laneid, 1)]
)


@T.prim_func
def snapshot_copy_linear_32_runtime_domain(
    source: T.Buffer((32,), "float16"),
    output: T.Buffer((32,), "float16"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    _lane = T.lane_id([32])
    Tx.copy(output[:], source[:], dispatch="fallback")


@T.prim_func
def snapshot_copy_variable_min_runtime_domain(
    source: T.Buffer((32,), "float16"),
    output: T.Buffer((32,), "float16"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    local: T.f16[1]
    local[0] = source[lane]
    Tx.copy(output[lane : lane + 1], local[:], dispatch="reg")


@T.prim_func
def snapshot_copy_cross_warp_extent_runtime_domain(
    source: T.Buffer((64, 2), "float32"),
    output: T.Buffer((64, 2), "float32"),
):
    T.device_entry()
    _warpgroup = T.warpgroup_id([1])
    warp = T.warp_id_in_wg([4])
    lane = T.lane_id([32])
    thread = T.meta_var(warp * 32 + lane)
    source_storage: T.f32[2]
    destination_storage: T.f32[2]
    source_view = source_storage.view(128, 2, layout=wg_local_layout(2))
    destination_view = destination_storage.view(128, 2, layout=wg_local_layout(2))
    if thread < 64:
        for column in T.serial(2):
            source_view[thread, column] = source[thread, column]
    Tx.wg.copy(destination_view[64:128, :], source_view[0:64, :])
    if thread >= 64:
        for column in T.serial(2):
            output[thread - 64, column] = destination_view[thread, column]


@T.prim_func
def snapshot_copy_transposed_ldstmatrix_runtime_domain(
    source_registers: T.Buffer((128, 8), "bfloat16"),
    output: T.Buffer((8, 128), "bfloat16"),
):
    T.device_entry()
    _warpgroup = T.warpgroup_id([1])
    warp = T.warp_id_in_wg([4])
    lane = T.lane_id([32])
    thread = T.meta_var(warp * 32 + lane)
    shared = T.alloc_buffer((8, 128), "bfloat16", scope="shared")
    source_storage = T.alloc_buffer((8,), "bfloat16", scope="local")
    source_view = source_storage.view(8, 128, layout=_SNAPSHOT_TRANSPOSED_LDSTMATRIX_LAYOUT)
    for register in T.serial(8):
        source_storage[register] = source_registers[thread, register]
    Tx.wg.copy(shared[:, :], source_view[:, :], dispatch="ldstmatrix")
    T.cuda.cta_sync()
    for row in T.serial(8):
        output[row, thread] = shared[row, thread]


@T.prim_func
def snapshot_copy_bound_ldstmatrix_runtime_domain(
    source_registers: T.Buffer((128, 4), "bfloat16"),
    output: T.Buffer((64, 64), "bfloat16"),
):
    T.device_entry()
    _warpgroup = T.warpgroup_id([1])
    warp = T.warp_id_in_wg([4])
    lane = T.lane_id([32])
    thread = T.meta_var(warp * 32 + lane)
    shared = T.alloc_buffer((64, 64), "bfloat16", scope="shared", align=16)
    source_storage = T.alloc_buffer((4,), "bfloat16", scope="local")
    source_view = source_storage.view(64, 8, layout=_SNAPSHOT_BOUND_LDSTMATRIX_LAYOUT)
    for register in T.serial(4):
        source_storage[register] = source_registers[thread, register]
    for block in T.unroll(8):
        column: T.let = block * 8
        Tx.wg.copy(shared[:, column : column + 8], source_view[:, :], dispatch="ldstmatrix")
    T.cuda.cta_sync()
    if thread < 64:
        for column in T.serial(64):
            output[thread, column] = shared[thread, column]


_RAW_TCGEN_LDST_LAYOUT = TileLayout(S[(128, 512) : (1 @ TLane, 1 @ TCol)])


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
    return tvm.script.from_source(
        "\n".join(
            (
                "@T.prim_func",
                f"def raw_tcgen_ldst_runtime_domain_{name}_{num}(output: T.Buffer((1,), 'int32')):",
                "    T.device_entry()",
                "    _cta = T.cta_id([1])",
                "    _warpgroup = T.warpgroup_id([1])",
                "    warp = T.warp_id_in_wg([4])",
                "    lane = T.lane_id([32])",
                "    row = T.meta_var(warp * 32 + lane)",
                "    tmem = T.decl_buffer((128, 512), 'uint32', scope='tmem', "
                "layout=TMEM_LAYOUT, allocated_addr=0)",
                "    registers = T.alloc_buffer((128,), 'uint32', scope='local')",
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
    return tvm.script.from_source(
        "\n".join(
            (
                "@T.prim_func",
                f"def raw_tcgen_cp_runtime_domain_{slug}(output: T.Buffer((1,), 'int32')):",
                "    T.device_entry()",
                "    _cluster = T.cluster_id([1])",
                "    cta = T.cta_id_in_cluster([2])",
                "    _warpgroup = T.warpgroup_id([1])",
                "    warp = T.warp_id_in_wg([4])",
                "    lane = T.lane_id([32])",
                "    thread = T.meta_var(warp * 32 + lane)",
                "    shared = T.alloc_buffer((16384,), 'uint8', scope='shared', align=128)",
                "    tmem = T.decl_buffer((128, 512), 'uint32', scope='tmem', "
                "layout=TMEM_LAYOUT, allocated_addr=0)",
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


def test_snapshot_copy_linear_32_runtime_domain_has_exact_oracle(tmp_path: Path):
    source = np.arange(32, dtype=np.float16) * np.float16(0.25) - np.float16(3)
    module = numsim.transpile(snapshot_copy_linear_32_runtime_domain, cache_dir=tmp_path)
    result = numsim.Engine(max_workers=1).run(
        module, {"source": source, "output": np.zeros_like(source)}
    )

    np.testing.assert_array_equal(result.outputs["output"], source)


def test_snapshot_copy_variable_min_runtime_domain_has_exact_oracle(tmp_path: Path):
    source = np.arange(32, dtype=np.float16) + np.float16(0.5)
    module = numsim.transpile(snapshot_copy_variable_min_runtime_domain, cache_dir=tmp_path)
    result = numsim.Engine(max_workers=1).run(
        module, {"source": source, "output": np.zeros_like(source)}
    )

    np.testing.assert_array_equal(result.outputs["output"], source)


def test_snapshot_copy_cross_warp_extent_runtime_domain_has_exact_oracle(tmp_path: Path):
    source = np.arange(128, dtype=np.float32).reshape(64, 2) + np.float32(0.25)
    module = numsim.transpile(snapshot_copy_cross_warp_extent_runtime_domain, cache_dir=tmp_path)
    result = numsim.Engine(max_workers=1).run(
        module, {"source": source, "output": np.zeros_like(source)}
    )

    np.testing.assert_array_equal(result.outputs["output"], source)


def test_snapshot_copy_transposed_ldstmatrix_runtime_domain_has_exact_oracle(tmp_path: Path):
    values = (np.arange(128 * 8, dtype=np.float32).reshape(128, 8) % 97) - np.float32(48)
    source = _bfloat16_bits(values)
    expected = _transposed_ldstmatrix_expected(source)
    module = numsim.transpile(
        snapshot_copy_transposed_ldstmatrix_runtime_domain, cache_dir=tmp_path
    )
    result = numsim.Engine(max_workers=1).run(
        module,
        {
            "source_registers": source,
            "output": np.zeros_like(expected),
        },
    )

    np.testing.assert_array_equal(result.outputs["output"], expected)


def test_snapshot_copy_bound_ldstmatrix_runtime_domain_has_exact_oracle(tmp_path: Path):
    values = (np.arange(128 * 4, dtype=np.float32).reshape(128, 4) % 89) - np.float32(44)
    source = _bfloat16_bits(values)
    expected = _bound_ldstmatrix_expected(source)
    module = numsim.transpile(snapshot_copy_bound_ldstmatrix_runtime_domain, cache_dir=tmp_path)
    result = numsim.Engine(max_workers=1).run(
        module,
        {
            "source_registers": source,
            "output": np.zeros_like(expected),
        },
    )

    np.testing.assert_array_equal(result.outputs["output"], expected)


def test_cuda_ldg_complete_finite_domain_executes_with_exact_physical_oracle(tmp_path: Path):
    module = numsim.transpile(CUDA_LDG_COMPLETE_RUNTIME_DOMAIN, cache_dir=tmp_path)
    arguments: dict[str, object] = {}
    expected: dict[str, np.ndarray] = {}
    for dtype in CUDA_LDG_DTYPES:
        name = dtype.replace("_", "")
        source, needs_binding = _storage(dtype, initialized=True)
        output, _ = _storage(dtype, initialized=False)
        arguments[f"source_{name}"] = source if needs_binding else source
        arguments[f"output_{name}"] = output if needs_binding else output
        expected[f"output_{name}"] = source.copy()

    result = numsim.Engine(max_workers=1).run(module, arguments)

    def check_every_dtype() -> None:
        assert len(expected) == 55
        assert set(result.outputs) == set(arguments)
        for name, value in expected.items():
            _assert_same_physical_value(result.outputs[name], value)

    check_every_dtype()


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
