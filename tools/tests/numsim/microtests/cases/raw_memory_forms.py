"""Paired GPU/NumSim cases for the raw `ld`/`st` PTX form coverage."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from tests.numsim.integration.test_raw_memory_artifact import (
    raw_global_nc_eviction_and_prefetch_hint_matrix,
    raw_global_nc_v2_destination_load,
    raw_global_v2_and_v8_destination_loads,
    raw_shared_v2_ordered_destination_loads,
    raw_sub_word_global_roundtrip,
    raw_sub_word_ordered_forms,
)

# Two spellings NumSim models but the TIRx CUDA backend cannot currently build,
# so they have NumSim-only coverage in
# `tests/numsim/integration/test_raw_memory_artifact.py` instead of a paired
# GPU case:
#
#   * `ld.relaxed` return-value form -- the emitted helper takes `(address)`
#     while the call site passes `(address, cache_policy)`
#     ("too many arguments in function call").
#   * `ld.volatile` destination-passing form -- the emitted helper takes
#     `(dst_ptr, src_ptr, cache_policy)` while the call site passes
#     `(dst_ptr, src_ptr)` ("too few arguments in function call").


@dataclass(frozen=True)
class RawMemoryFormCase:
    name: str
    prim_func: Any
    make_arguments: Callable[[], Mapping[str, Any]]
    outputs: tuple[str, ...] = ("output",)


def _source32(count: int) -> np.ndarray:
    return (np.arange(count, dtype=np.uint32) * np.uint32(0x01020408)) ^ np.uint32(0x5AA55AA5)


def _source64(count: int) -> np.ndarray:
    return (np.arange(count, dtype=np.uint64) * np.uint64(0x0102040810204080)) ^ np.uint64(
        0xA55A5AA5A55A5AA5
    )


RAW_MEMORY_FORM_CASES = (
    RawMemoryFormCase(
        "ld_v2_and_v8_destination_loads",
        raw_global_v2_and_v8_destination_loads,
        lambda: {
            "source32": _source32(256),
            "source64": _source64(64),
            "out_v2_32": np.zeros(64, dtype=np.uint32),
            "out_v8_32": np.zeros(256, dtype=np.uint32),
            "out_v2_64": np.zeros(64, dtype=np.uint64),
        },
        ("out_v2_32", "out_v8_32", "out_v2_64"),
    ),
    RawMemoryFormCase(
        "ld_v2_relaxed_and_acquire_destination_loads",
        raw_shared_v2_ordered_destination_loads,
        lambda: {
            "source": _source32(64),
            "relaxed": np.zeros(64, dtype=np.uint32),
            "acquired": np.zeros(64, dtype=np.uint32),
        },
        ("relaxed", "acquired"),
    ),
    RawMemoryFormCase(
        "ld_global_nc_v2_destination_load",
        raw_global_nc_v2_destination_load,
        lambda: {"source": _source32(64), "output": np.zeros(64, dtype=np.uint32)},
    ),
    RawMemoryFormCase(
        "sub_word_ld_st_extension",
        raw_sub_word_global_roundtrip,
        lambda: {
            "source": np.asarray(
                [(-128 + (index * 9) % 256) for index in range(32)], dtype=np.int32
            ),
            "out_b8": np.zeros(32, dtype=np.uint32),
            "out_s8": np.zeros(32, dtype=np.int32),
            "out_b16": np.zeros(32, dtype=np.uint16),
            "out_s16": np.zeros(32, dtype=np.int16),
            "out_b8_wide": np.zeros(32, dtype=np.uint16),
            "out_b8_pair": np.zeros((32, 2), dtype=np.uint16),
            "out_b16_wide": np.zeros(32, dtype=np.uint32),
            "out_s16_wide": np.zeros(32, dtype=np.int32),
            "out_u16_wide_store": np.zeros(32, dtype=np.uint32),
        },
        (
            "out_b8",
            "out_s8",
            "out_b16",
            "out_s16",
            "out_b8_wide",
            "out_b8_pair",
            "out_b16_wide",
            "out_s16_wide",
            "out_u16_wide_store",
        ),
    ),
    RawMemoryFormCase(
        "sub_word_ordered_ld_st_forms",
        raw_sub_word_ordered_forms,
        lambda: {
            "source": _source32(32),
            "acquired_u8": np.zeros(32, dtype=np.uint32),
            "acquired_u16": np.zeros(32, dtype=np.uint16),
            "volatile_s16": np.zeros(32, dtype=np.int16),
        },
        ("acquired_u8", "acquired_u16", "volatile_s16"),
    ),
    RawMemoryFormCase(
        "ld_global_nc_eviction_and_prefetch_hints",
        raw_global_nc_eviction_and_prefetch_hint_matrix,
        lambda: {"source": _source32(32), "output": np.zeros((32, 6), dtype=np.uint32)},
    ),
)


__all__ = ["RAW_MEMORY_FORM_CASES", "RawMemoryFormCase"]
