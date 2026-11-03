from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from tvm_tirx.script import tirx as T

_CP_SHAPE_ROUTES = (
    ("4x256b", "", 4, 32),
    ("32x128b", "warpx4", 32, 16),
    ("64x128b", "warpx2::01_23", 64, 16),
    ("64x128b", "warpx2::02_13", 64, 16),
    ("128x128b", "", 128, 16),
    ("128x256b", "", 128, 32),
)
_DECOMPRESS_MODES = ("", "b8x16.b4x16_p64", "b8x16.b6x16_p32")
_DECOMPRESS_NAMES = {"": "none", "b8x16.b4x16_p64": "b4", "b8x16.b6x16_p32": "b6"}
_LDST_SHAPES = ("32x32b", "16x32bx2", "16x64b", "16x128b", "16x256b")


def _physical_cp_source(atom_bytes: np.ndarray) -> np.ndarray:
    if atom_bytes.ndim != 3 or atom_bytes.shape[2] != 16:
        raise ValueError(f"expected [rows, atoms, 16], got {atom_bytes.shape}")
    rows, atoms, _ = atom_bytes.shape
    physical = np.zeros(((rows + 7) // 8) * atoms * 128, dtype=np.uint8)
    for row in range(rows):
        for atom in range(atoms):
            offset = (row // 8) * atoms * 128 + atom * 128 + (row % 8) * 16
            physical[offset : offset + 16] = atom_bytes[row, atom]
    return physical


def _cp_inputs(rows: int, row_bytes: int, decompress: str) -> tuple[np.ndarray, np.ndarray]:
    row = np.arange(rows, dtype=np.uint32)[:, None]
    column = np.arange(row_bytes, dtype=np.uint32)[None, :]
    if decompress == "":
        logical = ((row * 37 + column * 11 + 3) % 251 + 1).astype(np.uint8)
        atoms = logical.reshape(rows, row_bytes // 16, 16)
        return _physical_cp_source(atoms), logical

    limit = 16 if decompress == "b8x16.b4x16_p64" else 64
    shift = 2 if limit == 16 else 0
    codes = ((row * 5 + column * 3 + 1) % limit).astype(np.uint8)
    atoms = np.zeros((rows, row_bytes // 16, 16), dtype=np.uint8)
    for logical_row in range(rows):
        for atom in range(row_bytes // 16):
            values = codes[logical_row, atom * 16 : (atom + 1) * 16]
            if limit == 16:
                atoms[logical_row, atom, :8] = values[0::2] | (values[1::2] << np.uint8(4))
            else:
                packed = sum(int(value) << (6 * index) for index, value in enumerate(values))
                atoms[logical_row, atom, :12] = np.frombuffer(
                    packed.to_bytes(12, "little"), dtype=np.uint8
                )
    return _physical_cp_source(atoms), (codes << np.uint8(shift)).astype(np.uint8)


def _make_raw_cp_kernel(
    shape: str,
    multicast: str,
    rows: int,
    row_bytes: int,
    cta_group: int,
    physical_size: int,
):
    atoms = row_bytes // 16
    ldo = 0 if atoms == 1 else 8
    sdo = 1 if shape == "4x256b" else atoms * 8
    cp_chain = f"tcgen05.cp.cta_group::{cta_group}.{shape}"
    if multicast:
        cp_chain = f"{cp_chain}.{multicast}"
    cp_b4_chain = f"{cp_chain}.b8x16.b4x16_p64"
    cp_b6_chain = f"{cp_chain}.b8x16.b6x16_p32"

    @T.function
    def kernel(
        source_none: T.Tensor((physical_size,), "uint8"),
        source_b4: T.Tensor((physical_size,), "uint8"),
        source_b6: T.Tensor((physical_size,), "uint8"),
        output: T.Tensor((3, 2, 128, 8), "uint32"),
    ):
        T.device_entry(launch=T.cuda.LaunchConfig(grid=2, block=128, cluster=2))
        cta = T.cuda.cluster_cta_id("x")
        warp = T.cuda.warp_in_warpgroup()
        lane = T.cuda.lane_id()
        thread = T.meta_var(warp * 32 + lane)
        address = T.alloc_shared((1,), "uint32")
        barrier = T.alloc_shared((1,), "uint64")
        shared_none = T.alloc_shared((physical_size,), "uint8", align=128)
        shared_b4 = T.alloc_shared((physical_size,), "uint8", align=128)
        shared_b6 = T.alloc_shared((physical_size,), "uint8", align=128)
        registers = T.alloc_local((8,), "uint32")
        desc_none: T.uint64
        desc_b4: T.uint64
        desc_b6: T.uint64

        for copy_index in T.serial((physical_size + 127) // 128):
            offset = thread + copy_index * 128
            if offset < physical_size:
                shared_none[offset] = source_none[offset]
                shared_b4[offset] = source_b4[offset]
                shared_b6[offset] = source_b6[offset]

        if warp == 0:
            T.ptx[f"tcgen05.alloc.cta_group::{cta_group}.sync.aligned.shared::cta.b32"](
                T.address_of(address[0]), 32
            )
            if lane == 0:
                T.ptx.mbarrier.init.shared.b64(T.address_of(barrier[0]), 1)
        T.ptx.fence.proxy.async_.shared__cta()
        T.ptx.fence.mbarrier_init.release.cluster()
        T.cuda.cluster_sync()

        for register in T.unroll(8):
            registers[register] = T.uint32(0)
        T.ptx["tcgen05.st.sync.aligned.32x32b.x8.b32"](
            address[0],
            registers[0],
            registers[1],
            registers[2],
            registers[3],
            registers[4],
            registers[5],
            registers[6],
            registers[7],
        )
        T.ptx["tcgen05.st.sync.aligned.32x32b.x8.b32"](
            T.cuda.get_tmem_addr(address[0], 0, 8),
            registers[0],
            registers[1],
            registers[2],
            registers[3],
            registers[4],
            registers[5],
            registers[6],
            registers[7],
        )
        T.ptx["tcgen05.st.sync.aligned.32x32b.x8.b32"](
            T.cuda.get_tmem_addr(address[0], 0, 16),
            registers[0],
            registers[1],
            registers[2],
            registers[3],
            registers[4],
            registers[5],
            registers[6],
            registers[7],
        )
        T.ptx.tcgen05.wait__st.sync.aligned()
        T.cuda.cluster_sync()

        if ((cta_group == 1) or (cta == 0)) and warp == 0 and lane == 0:
            T.cuda.tcgen05.encode_matrix_descriptor(
                T.address_of(desc_none), T.address_of(shared_none[0]), ldo, sdo, 0
            )
            T.cuda.tcgen05.encode_matrix_descriptor(
                T.address_of(desc_b4), T.address_of(shared_b4[0]), ldo, sdo, 0
            )
            T.cuda.tcgen05.encode_matrix_descriptor(
                T.address_of(desc_b6), T.address_of(shared_b6[0]), ldo, sdo, 0
            )
            T.ptx[cp_chain](address[0], desc_none)
            T.ptx[cp_b4_chain](address[0] + T.uint32(8), desc_b4)
            T.ptx[cp_b6_chain](address[0] + T.uint32(16), desc_b6)
            T.ptx[
                f"tcgen05.commit.cta_group::{cta_group}.mbarrier::arrive::one.shared::cluster.b64"
            ](T.address_of(barrier[0]))
        if ((cta_group == 1) or (cta == 0)) and warp == 0:
            T.cuda.mbarrier_wait(T.address_of(barrier[0]), 0)
        T.cuda.cluster_sync()

        for mode in T.unroll(3):
            T.ptx["tcgen05.ld.sync.aligned.32x32b.x8.b32"](
                registers[0],
                registers[1],
                registers[2],
                registers[3],
                registers[4],
                registers[5],
                registers[6],
                registers[7],
                T.cuda.get_tmem_addr(address[0], 0, mode * 8),
            )
            T.ptx.tcgen05.wait__ld.sync.aligned()
            for register in T.unroll(8):
                output[mode, cta, thread, register] = registers[register]
        T.cuda.cluster_sync()

        if warp == 0:
            T.ptx[f"tcgen05.dealloc.cta_group::{cta_group}.sync.aligned.b32"](address[0], 32)
            T.ptx[f"tcgen05.relinquish_alloc_permit.cta_group::{cta_group}.sync.aligned"]()

    return kernel


def _ldst_location(
    shape: str, num: int, packed: bool, warp: int, lane: int, register: int, split_padding: int = 0
) -> tuple[int, int]:
    if shape == "32x32b":
        row, column = warp * 32 + lane, register
    elif shape == "16x32bx2":
        width = 2 if packed else 1
        return warp * 32 + lane % 16, register * width + (lane // 16) * (
            num * width + split_padding
        )
    elif shape == "16x64b":
        row = warp * 32 + (lane >> 2) + 8 * (lane & 1)
        column = ((lane >> 1) & 1) + 2 * register
    elif shape == "16x128b":
        row = warp * 32 + (lane >> 2) + 8 * (register & 1)
        column = (lane & 3) + 4 * (register >> 1)
    elif shape == "16x256b":
        row = warp * 32 + (lane >> 2) + 8 * ((register >> 1) & 1)
        column = (register & 1) + 2 * (lane & 3) + 8 * (register >> 2)
    else:
        raise ValueError(shape)
    return row, column * (2 if packed else 1)


def _pack_cell(low: int, high: int) -> int:
    return (low & 0xFFFF) | ((high & 0xFFFF) << 16)


def _ldst_oracle(shape: str, packed: bool, split_padding: int = 0) -> dict[str, np.ndarray]:
    seed = np.fromfunction(
        lambda row, col: 0x10000000 + row * 0x100 + col,
        (128, 32),
        dtype=np.uint32,
    ).astype(np.uint32)
    factor = {"32x32b": 1, "16x32bx2": 1, "16x64b": 1, "16x128b": 2, "16x256b": 4}[shape]
    loads = np.zeros((2, 4, 32, 8), dtype=np.uint32)
    stores = np.zeros((2, 128, 32), dtype=np.uint32)
    for count_index, num in enumerate((1, 2)):
        register_count = factor * num
        for warp in range(4):
            for lane in range(32):
                for register in range(register_count):
                    row, column = _ldst_location(
                        shape, num, packed, warp, lane, register, split_padding
                    )
                    if packed:
                        loads[count_index, warp, lane, register] = _pack_cell(
                            int(seed[row, column]), int(seed[row, column + 1])
                        )
                    else:
                        loads[count_index, warp, lane, register] = seed[row, column]

                    value = np.uint32(
                        0x20000000
                        + count_index * 0x01000000
                        + warp * 0x00100000
                        + lane * 0x100
                        + register
                    )
                    if packed:
                        stores[count_index, row, column] = value & np.uint32(0xFFFF)
                        stores[count_index, row, column + 1] = value >> np.uint32(16)
                    else:
                        stores[count_index, row, column] = value
    return {"loads": loads, "stores": stores}


def _make_raw_ldst_kernel(shape: str, packed: bool, split_padding: int = 0):
    factor = {"32x32b": 1, "16x32bx2": 1, "16x64b": 1, "16x128b": 2, "16x256b": 4}[shape]
    max_registers = factor * 2
    split_x1 = ((2 if packed else 1) + split_padding,) if shape == "16x32bx2" else ()
    split_x2 = ((4 if packed else 2) + split_padding,) if shape == "16x32bx2" else ()

    @T.function
    def kernel(
        load_output: T.Tensor((2, 4, 32, 8), "uint32"),
        store_output: T.Tensor((2, 128, 32), "uint32"),
    ):
        T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=128))
        warp = T.cuda.warp_in_warpgroup()
        lane = T.cuda.lane_id()
        thread = T.meta_var(warp * 32 + lane)
        address = T.alloc_shared((1,), "uint32")
        registers = T.alloc_local((32,), "uint32")
        target = T.alloc_local((8,), "uint32")

        if warp == 0:
            T.ptx.tcgen05.alloc.cta_group__1.sync.aligned.shared__cta.b32(
                T.address_of(address[0]), 64
            )
        T.cuda.cta_sync()

        for column in T.unroll(32):
            registers[column] = T.cast(0x10000000 + thread * 0x100 + column, "uint32")
        T.ptx["tcgen05.st.sync.aligned.32x32b.x32.b32"](
            address[0],
            registers[0],
            registers[1],
            registers[2],
            registers[3],
            registers[4],
            registers[5],
            registers[6],
            registers[7],
            registers[8],
            registers[9],
            registers[10],
            registers[11],
            registers[12],
            registers[13],
            registers[14],
            registers[15],
            registers[16],
            registers[17],
            registers[18],
            registers[19],
            registers[20],
            registers[21],
            registers[22],
            registers[23],
            registers[24],
            registers[25],
            registers[26],
            registers[27],
            registers[28],
            registers[29],
            registers[30],
            registers[31],
        )
        T.ptx.tcgen05.wait__st.sync.aligned()
        T.cuda.cta_sync()

        if T.constexpr(factor == 1):
            T.ptx[f"tcgen05.ld.sync.aligned.{shape}.x1{'.pack::16b' if packed else ''}.b32"](
                target[0], address[0], *split_x1
            )
        elif T.constexpr(factor == 2):
            T.ptx[f"tcgen05.ld.sync.aligned.{shape}.x1{'.pack::16b' if packed else ''}.b32"](
                target[0],
                target[1],
                address[0],
                *split_x1,
            )
        else:
            T.ptx[f"tcgen05.ld.sync.aligned.{shape}.x1{'.pack::16b' if packed else ''}.b32"](
                target[0],
                target[1],
                target[2],
                target[3],
                address[0],
                *split_x1,
            )
        T.ptx.tcgen05.wait__ld.sync.aligned()
        for register in T.unroll(8):
            load_output[0, warp, lane, register] = T.if_then_else(
                register < factor, target[register], T.uint32(0)
            )

        if T.constexpr(max_registers == 2):
            T.ptx[f"tcgen05.ld.sync.aligned.{shape}.x2{'.pack::16b' if packed else ''}.b32"](
                target[0],
                target[1],
                address[0],
                *split_x2,
            )
        elif T.constexpr(max_registers == 4):
            T.ptx[f"tcgen05.ld.sync.aligned.{shape}.x2{'.pack::16b' if packed else ''}.b32"](
                target[0],
                target[1],
                target[2],
                target[3],
                address[0],
                *split_x2,
            )
        else:
            T.ptx[f"tcgen05.ld.sync.aligned.{shape}.x2{'.pack::16b' if packed else ''}.b32"](
                target[0],
                target[1],
                target[2],
                target[3],
                target[4],
                target[5],
                target[6],
                target[7],
                address[0],
                *split_x2,
            )
        T.ptx.tcgen05.wait__ld.sync.aligned()
        for register in T.unroll(8):
            load_output[1, warp, lane, register] = T.if_then_else(
                register < max_registers, target[register], T.uint32(0)
            )

        for count_index in T.unroll(2):
            for column in T.unroll(32):
                registers[column] = T.uint32(0)
            T.ptx["tcgen05.st.sync.aligned.32x32b.x32.b32"](
                address[0],
                registers[0],
                registers[1],
                registers[2],
                registers[3],
                registers[4],
                registers[5],
                registers[6],
                registers[7],
                registers[8],
                registers[9],
                registers[10],
                registers[11],
                registers[12],
                registers[13],
                registers[14],
                registers[15],
                registers[16],
                registers[17],
                registers[18],
                registers[19],
                registers[20],
                registers[21],
                registers[22],
                registers[23],
                registers[24],
                registers[25],
                registers[26],
                registers[27],
                registers[28],
                registers[29],
                registers[30],
                registers[31],
            )
            T.ptx.tcgen05.wait__st.sync.aligned()
            T.cuda.cta_sync()

            for register in T.unroll(8):
                target[register] = T.cast(
                    0x20000000
                    + count_index * 0x01000000
                    + warp * 0x00100000
                    + lane * 0x100
                    + register,
                    "uint32",
                )
            if T.constexpr(factor == 1):
                if count_index == 0:
                    T.ptx[
                        f"tcgen05.st.sync.aligned.{shape}.x1{'.unpack::16b' if packed else ''}.b32"
                    ](
                        address[0],
                        *split_x1,
                        target[0],
                    )
                else:
                    T.ptx[
                        f"tcgen05.st.sync.aligned.{shape}.x2{'.unpack::16b' if packed else ''}.b32"
                    ](
                        address[0],
                        *split_x2,
                        target[0],
                        target[1],
                    )
            elif T.constexpr(factor == 2):
                if count_index == 0:
                    T.ptx[
                        f"tcgen05.st.sync.aligned.{shape}.x1{'.unpack::16b' if packed else ''}.b32"
                    ](
                        address[0],
                        *split_x1,
                        target[0],
                        target[1],
                    )
                else:
                    T.ptx[
                        f"tcgen05.st.sync.aligned.{shape}.x2{'.unpack::16b' if packed else ''}.b32"
                    ](
                        address[0],
                        *split_x2,
                        target[0],
                        target[1],
                        target[2],
                        target[3],
                    )
            else:
                if count_index == 0:
                    T.ptx[
                        f"tcgen05.st.sync.aligned.{shape}.x1{'.unpack::16b' if packed else ''}.b32"
                    ](
                        address[0],
                        *split_x1,
                        target[0],
                        target[1],
                        target[2],
                        target[3],
                    )
                else:
                    T.ptx[
                        f"tcgen05.st.sync.aligned.{shape}.x2{'.unpack::16b' if packed else ''}.b32"
                    ](
                        address[0],
                        *split_x2,
                        target[0],
                        target[1],
                        target[2],
                        target[3],
                        target[4],
                        target[5],
                        target[6],
                        target[7],
                    )
            T.ptx.tcgen05.wait__st.sync.aligned()
            T.cuda.cta_sync()
            T.ptx["tcgen05.ld.sync.aligned.32x32b.x32.b32"](
                registers[0],
                registers[1],
                registers[2],
                registers[3],
                registers[4],
                registers[5],
                registers[6],
                registers[7],
                registers[8],
                registers[9],
                registers[10],
                registers[11],
                registers[12],
                registers[13],
                registers[14],
                registers[15],
                registers[16],
                registers[17],
                registers[18],
                registers[19],
                registers[20],
                registers[21],
                registers[22],
                registers[23],
                registers[24],
                registers[25],
                registers[26],
                registers[27],
                registers[28],
                registers[29],
                registers[30],
                registers[31],
                address[0],
            )
            T.ptx.tcgen05.wait__ld.sync.aligned()
            for column in T.unroll(32):
                store_output[count_index, thread, column] = registers[column]
            T.cuda.cta_sync()

        if warp == 0:
            T.ptx.tcgen05.dealloc.cta_group__1.sync.aligned.b32(address[0], 64)
            T.ptx.tcgen05.relinquish_alloc_permit.cta_group__1.sync.aligned()

    return kernel


def _tile_cp_input(dtype: str, bits: int) -> tuple[np.ndarray, np.ndarray]:
    elements = 128 // bits
    if bits == 8:
        host = (
            ((np.arange(32 * elements, dtype=np.uint16) * 13 + 5) % 251)
            .astype(np.uint8)
            .reshape(32, elements)
        )
        bytes_ = host
    elif bits == 16:
        host = (np.arange(32 * elements, dtype=np.float32).reshape(32, elements) - 100).astype(
            np.float16
        )
        bytes_ = np.ascontiguousarray(host).view(np.uint8).reshape(32, 16)
    else:
        host = np.arange(32 * elements, dtype=np.uint32).reshape(32, elements) * np.uint32(
            0x01010101
        ) + np.uint32(0x10203040)
        bytes_ = np.ascontiguousarray(host).view(np.uint8).reshape(32, 16)
    expected = np.tile(np.ascontiguousarray(bytes_).view(np.uint32).reshape(32, 4), (4, 1))
    return host, expected


@dataclass(frozen=True)
class Tcgen05TransferCase:
    name: str
    prim_func: Any
    make_arguments: Callable[[], Mapping[str, Any]]
    outputs: tuple[str, ...]


def _transfer_cases() -> list[Tcgen05TransferCase]:
    cases: list[Tcgen05TransferCase] = []
    for shape, multicast, rows, row_bytes in _CP_SHAPE_ROUTES:
        inputs = {mode: _cp_inputs(rows, row_bytes, mode) for mode in _DECOMPRESS_MODES}
        physical_size = len(inputs[""][0])
        for cta_group in (1, 2):
            kernel = _make_raw_cp_kernel(
                shape, multicast, rows, row_bytes, cta_group, physical_size
            )

            def make_raw_cp_arguments(inputs=inputs):
                return {
                    "source_none": inputs[""][0],
                    "source_b4": inputs["b8x16.b4x16_p64"][0],
                    "source_b6": inputs["b8x16.b6x16_p32"][0],
                    "output": np.zeros((3, 2, 128, 8), dtype=np.uint32),
                }

            cases.append(
                Tcgen05TransferCase(
                    f"cp_{shape}_{multicast or 'direct'}_cta{cta_group}",
                    kernel,
                    make_raw_cp_arguments,
                    ("output",),
                )
            )

    for shape in _LDST_SHAPES:
        for packed in (False, True):
            for padding in (0, 6) if shape == "16x32bx2" else (0,):
                cases.append(
                    Tcgen05TransferCase(
                        f"ldst_{shape}_{'pack16' if packed else 'plain'}"
                        + (f"_gap{padding}" if padding else ""),
                        _make_raw_ldst_kernel(shape, packed, padding),
                        lambda: {
                            "load_output": np.zeros((2, 4, 32, 8), dtype=np.uint32),
                            "store_output": np.zeros((2, 128, 32), dtype=np.uint32),
                        },
                        ("load_output", "store_output"),
                    )
                )

    return cases


TCGEN05_TRANSFER_CASES = tuple(_transfer_cases())
