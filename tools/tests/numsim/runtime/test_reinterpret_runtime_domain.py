from __future__ import annotations

import itertools
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
import tvm
from tvm.script import tirx as T

from tirx_harness import numsim
from tirx_harness.numsim.transpiler.frontend import analyze


_SCALAR_BITS = {
    "bool": 8,
    "int8": 8,
    "uint8": 8,
    "float8_e3m4": 8,
    "float8_e4m3": 8,
    "float8_e4m3b11fnuz": 8,
    "float8_e4m3fn": 8,
    "float8_e4m3fnuz": 8,
    "float8_e5m2": 8,
    "float8_e5m2fnuz": 8,
    "float8_e8m0fnu": 8,
    "int16": 16,
    "uint16": 16,
    "float16": 16,
    "bfloat16": 16,
    "int32": 32,
    "uint32": 32,
    "float32": 32,
    "int64": 64,
    "uint64": 64,
    "float64": 64,
}
_RAW_SCALARS = frozenset(
    {
        "int8",
        "uint8",
        "int16",
        "uint16",
        "int32",
        "uint32",
        "int64",
        "uint64",
        "float32",
        "float64",
    }
)
_DIRECT_IDENTITY_DTYPES = ("bool", "float16", "bfloat16")
_CANONICAL_DTYPE = {8: "uint8", 16: "uint16", 32: "uint32", 64: "uint64", 128: "uint64x2"}
_CANONICAL_NUMPY_DTYPE = {
    8: np.uint8,
    16: np.uint16,
    32: np.uint32,
    64: np.uint64,
}
_SCALAR_NUMPY_DTYPE = {
    "int8": np.int8,
    "uint8": np.uint8,
    "int16": np.int16,
    "uint16": np.uint16,
    "int32": np.int32,
    "uint32": np.uint32,
    "int64": np.int64,
    "uint64": np.uint64,
    "float32": np.float32,
    "float64": np.float64,
}


def _storage_dtypes() -> dict[str, int]:
    result = dict(_SCALAR_BITS)
    for element, element_bits in _SCALAR_BITS.items():
        if element == "bool":
            continue
        for total_bits in (16, 32, 64, 128):
            if total_bits % element_bits:
                continue
            lanes = total_bits // element_bits
            if lanes > 1:
                result[f"{element}x{lanes}"] = total_bits
    return result


_STORAGE_DTYPES = _storage_dtypes()
_RAW_DTYPES_BY_WIDTH = {
    bits: tuple(
        sorted(
            dtype
            for dtype, dtype_bits in _STORAGE_DTYPES.items()
            if dtype_bits == bits and (dtype in _RAW_SCALARS or "x" in dtype)
        )
    )
    for bits in (8, 16, 32, 64, 128)
}


@dataclass(frozen=True)
class _ReinterpretShard:
    name: str
    kind: str
    bit_width: int | None
    signatures: tuple[tuple[str, str], ...]


_REINTERPRET_SHARDS = (
    *(
        _ReinterpretShard(
            name=f"raw-{bits}",
            kind="raw",
            bit_width=bits,
            signatures=tuple(itertools.product(dtypes, repeat=2)),
        )
        for bits, dtypes in _RAW_DTYPES_BY_WIDTH.items()
    ),
    _ReinterpretShard(
        name="scalar-identities",
        kind="identity",
        bit_width=None,
        signatures=tuple((dtype, dtype) for dtype in _DIRECT_IDENTITY_DTYPES),
    ),
    _ReinterpretShard(
        name="handle-token",
        kind="handle",
        bit_width=None,
        signatures=(("handle", "uint64"), ("uint64", "handle")),
    ),
)


@T.prim_func
def reinterpret_float16_payload(
    source_bits: T.Buffer((32,), "uint16"), output: T.Buffer((32,), "float32")
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    value: T.float16 = T.reinterpret("float16", source_bits[lane])
    output[lane] = T.cast(value, "float32")


@T.prim_func
def reinterpret_low_precision_payloads(
    source_bits: T.Buffer((32,), "uint16"),
    f16_bits: T.Buffer((32,), "uint16"),
    bf16_bits: T.Buffer((32,), "uint16"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    f16_value: T.float16 = T.reinterpret("float16", source_bits[lane])
    bf16_value: T.bfloat16 = T.reinterpret("bfloat16", source_bits[lane])
    f16_bits[lane] = T.reinterpret("uint16", f16_value)
    bf16_bits[lane] = T.reinterpret("uint16", bf16_value)


def _identifier(dtype: str) -> str:
    return re.sub(r"[^0-9A-Za-z_]", "_", dtype)


def _make_raw_kernel(shard: _ReinterpretShard):
    assert shard.bit_width is not None
    sources = sorted({source for source, _target in shard.signatures})
    source_index = {dtype: index for index, dtype in enumerate(sources)}
    parameters = [
        f'    source_{source_index[dtype]}: T.Buffer((1,), "{dtype}"),' for dtype in sources
    ]
    parameters.append(
        f'    output_bits: T.Buffer(({len(shard.signatures)},), "{_CANONICAL_DTYPE[shard.bit_width]}"),'
    )
    statements: list[str] = []
    for index, (source, target) in enumerate(shard.signatures):
        statements.extend(
            (
                f'        value_{index}: T.let = T.call_intrin("{target}", "tirx.reinterpret", source_{source_index[source]}[0])',
                f'        output_bits[{index}] = T.call_intrin("{_CANONICAL_DTYPE[shard.bit_width]}", "tirx.reinterpret", value_{index})',
            )
        )
    return tvm.script.from_source(
        "\n".join(
            (
                "@T.prim_func",
                f"def reinterpret_{shard.name.replace('-', '_')}(",
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


def _make_identity_kernel(shard: _ReinterpretShard):
    parameters: list[str] = []
    statements: list[str] = []
    for dtype, _target in shard.signatures:
        suffix = _identifier(dtype)
        parameters.extend(
            (
                f'    source_{suffix}: T.Buffer((1,), "{dtype}"),',
                f'    output_{suffix}: T.Buffer((1,), "{dtype}"),',
            )
        )
        statements.append(
            f'        output_{suffix}[0] = T.call_intrin("{dtype}", "tirx.reinterpret", source_{suffix}[0])'
        )
    return tvm.script.from_source(
        "\n".join(
            (
                "@T.prim_func",
                "def reinterpret_scalar_identities(",
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


def _make_handle_kernel():
    return tvm.script.from_source(
        "\n".join(
            (
                "@T.prim_func",
                "def reinterpret_handle_token(",
                '    source: T.Buffer((1,), "uint32"),',
                '    output: T.Buffer((1,), "uint32"),',
                "):",
                "    T.device_entry()",
                "    _warp = T.warp_id([1])",
                "    lane = T.lane_id([32])",
                "    if lane == 0:",
                '        token: T.uint64 = T.call_intrin("uint64", "tirx.reinterpret", source.ptr_to([0]))',
                '        pointer: T.let[T.handle] = T.call_intrin("handle", "tirx.reinterpret", token)',
                "        T.ptx.ld.global_.u32(output[0], pointer)",
            )
        ),
        extra_vars={"T": T},
    )


def _make_kernel(shard: _ReinterpretShard):
    if shard.kind == "raw":
        return _make_raw_kernel(shard)
    if shard.kind == "identity":
        return _make_identity_kernel(shard)
    assert shard.kind == "handle"
    return _make_handle_kernel()


def _raw_pattern(bit_width: int, index: int) -> bytes:
    if bit_width == 8:
        value = 0x41 + index
    elif bit_width == 16:
        value = 0x3400 | ((index + 1) * 0x21)
    elif bit_width == 32:
        value = 0x3F000000 | (((index + 1) * 0x10203) & 0x007FFFFF)
    elif bit_width == 64:
        low = 0x3F000000 | (((index + 1) * 0x10203) & 0x007FFFFF)
        high = 0x3FF00000 | (((index + 1) * 0x51) & 0x000FFFFF)
        value = low | (high << 32)
    else:
        assert bit_width == 128
        words = (
            0x3F000000 | (((index + 1) * 0x10203) & 0x007FFFFF),
            0x3FF00000 | (((index + 1) * 0x51) & 0x000FFFFF),
            0x40000000 | (((index + 1) * 0x20305) & 0x007FFFFF),
            0x40080000 | (((index + 1) * 0x71) & 0x0007FFFF),
        )
        value = sum(word << (32 * offset) for offset, word in enumerate(words))
    return value.to_bytes(bit_width // 8, byteorder="little", signed=False)


def _source_array(dtype: str, raw: bytes) -> np.ndarray:
    if "x" in dtype:
        return np.frombuffer(raw, dtype=np.dtype(f"V{len(raw)}")).copy()
    return np.frombuffer(raw, dtype=_SCALAR_NUMPY_DTYPE[dtype]).copy()


def _raw_arguments(shard: _ReinterpretShard) -> tuple[dict[str, object], np.ndarray]:
    assert shard.bit_width is not None
    sources = sorted({source for source, _target in shard.signatures})
    source_index = {dtype: index for index, dtype in enumerate(sources)}
    source_bytes = {
        dtype: _raw_pattern(shard.bit_width, index) for index, dtype in enumerate(sources)
    }
    arguments: dict[str, object] = {
        f"source_{source_index[dtype]}": _source_array(dtype, source_bytes[dtype])
        for dtype in sources
    }
    if shard.bit_width == 128:
        output = np.zeros(len(shard.signatures), dtype=np.dtype("V16"))
        arguments["output_bits"] = output
        expected = np.frombuffer(
            b"".join(source_bytes[source] for source, _target in shard.signatures),
            dtype=np.dtype("V16"),
        ).copy()
    else:
        dtype = _CANONICAL_NUMPY_DTYPE[shard.bit_width]
        arguments["output_bits"] = np.zeros(len(shard.signatures), dtype=dtype)
        expected = np.asarray(
            [
                int.from_bytes(source_bytes[source], byteorder="little", signed=False)
                for source, _target in shard.signatures
            ],
            dtype=dtype,
        )
    return arguments, expected


def _identity_arguments() -> tuple[dict[str, object], dict[str, np.ndarray]]:
    sources = {
        "bool": np.asarray([True], dtype=np.bool_),
        "float16": np.asarray([0x3555], dtype=np.uint16).view(np.float16),
        "bfloat16": np.asarray([0xBFC1], dtype=np.uint16),
    }
    arguments: dict[str, object] = {}
    expected: dict[str, np.ndarray] = {}
    for dtype, source in sources.items():
        suffix = _identifier(dtype)
        arguments[f"source_{suffix}"] = source
        output = np.zeros_like(source)
        arguments[f"output_{suffix}"] = output if dtype == "bfloat16" else output
        expected[f"output_{suffix}"] = source.view(np.uint8).copy()
    return arguments, expected


def _assert_identity_bytes(outputs: dict[str, object], expected: dict[str, np.ndarray]) -> None:
    for name, expected_bytes in expected.items():
        np.testing.assert_array_equal(np.asarray(outputs[name]).view(np.uint8), expected_bytes)


@pytest.mark.parametrize("shard", _REINTERPRET_SHARDS, ids=lambda shard: shard.name)
def test_complete_reinterpret_public_domain_preserves_physical_bits(
    tmp_path: Path, shard: _ReinterpretShard
):
    kernel = _make_kernel(shard)
    spec = analyze(kernel)
    assert spec.unsupported == ()
    module = numsim.transpile(kernel, cache_dir=tmp_path)

    if shard.kind == "raw":
        arguments, expected = _raw_arguments(shard)
        result = numsim.Engine(max_workers=1).run(module, arguments)

        def check() -> None:
            np.testing.assert_array_equal(result.outputs["output_bits"], expected)

    elif shard.kind == "identity":
        arguments, expected = _identity_arguments()
        result = numsim.Engine(max_workers=1).run(module, arguments)

        def check() -> None:
            _assert_identity_bytes(result.outputs, expected)

    else:
        source = np.asarray([0xA5C31F07], dtype=np.uint32)
        result = numsim.Engine(max_workers=1).run(
            module,
            {
                "source": source,
                "output": np.zeros(1, dtype=np.uint32),
            },
        )

        def check() -> None:
            np.testing.assert_array_equal(result.outputs["output"], source)

    check()


def test_uint16_payload_reinterpret_decodes_float16_values(tmp_path):
    source_bits = np.array(
        [
            0x0000,
            0x8000,
            0x0001,
            0x03FF,
            0x0400,
            0x3555,
            0x3C00,
            0xC100,
            0x7BFF,
            0x7C00,
            0xFC00,
        ],
        dtype=np.uint16,
    )
    source_bits = np.resize(source_bits, 32)
    result = numsim.Engine(max_workers=1).run(
        numsim.transpile(reinterpret_float16_payload, cache_dir=tmp_path),
        {"source_bits": source_bits, "output": np.zeros(32, dtype=np.float32)},
    )

    expected = source_bits.view(np.float16).astype(np.float32)
    np.testing.assert_array_equal(
        result.outputs["output"].view(np.uint32), expected.view(np.uint32)
    )


def test_varying_low_precision_reinterpret_preserves_payloads(tmp_path):
    # Finite values and signed zeros in both formats, including subnormals.
    bits = np.resize(
        np.array(
            [
                0x0000,
                0x8000,
                0x0001,
                0x8001,
                0x007F,
                0x0080,
                0x03FF,
                0x0400,
                0x3555,
                0x3C00,
                0x4000,
                0x7BFF,
                0x807F,
                0x8400,
                0xBC00,
                0xFBFF,
            ],
            dtype=np.uint16,
        ),
        32,
    )
    module = numsim.transpile(
        reinterpret_low_precision_payloads,
        cache_dir=tmp_path,
    )
    result = numsim.Engine(max_workers=1).run(
        module,
        {
            "source_bits": bits,
            "f16_bits": np.zeros(32, dtype=np.uint16),
            "bf16_bits": np.zeros(32, dtype=np.uint16),
        },
    )
    np.testing.assert_array_equal(result.outputs["f16_bits"], bits)
    np.testing.assert_array_equal(result.outputs["bf16_bits"], bits)
