"""Native Synccheck runtime evidence for protocol-bearing CUDA/PTX payload calls."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from tests.numsim.microtests.cases.mma_sync import (
    make_dense_f16_case,
    make_sparse_f16_case,
    raw_dense_f16_f32_k16,
    raw_sparse_f16_f32_k32,
)
from tests.numsim.microtests.cases.tcgen05_advanced_mma import (
    tcgen05_block_scaled_mxf4,
)
from tests.numsim.microtests.cases.tcgen05_lifecycle_ldst import (
    tcgen05_bf16_mma,
    tcgen05_cp_warpx4,
)
from tests.numsim.runtime.test_matrix_instruction_codegen import (
    mma_fragment_fill_and_store,
    ptx_mma_legacy_f16_m16n8k16,
)
from tests.numsim.runtime.test_memory_ops import (
    _tensor_map,
    raw_tma_gather4_bar_address,
    raw_tma_prefetch,
    raw_tma_reduce_add,
)
from tests.numsim.runtime.test_non_tensor_bulk_forms import raw_bulk_prefetch
from tests.numsim.runtime.test_ptx_register_bits import (
    ptx_half_abs_and_setp,
    ptx_predicate_data_path,
    ptx_selp_b16_bits,
    ptx_typed_move_and_b16_pack,
)
from tests.numsim.runtime.test_scalar_control import pointer_conversions_and_descriptor
from tests.numsim.support.kernels import raw_tma_roundtrip
from tests.numsim.support.manifest import call_op_names
from tirx_harness import numsim
from tirx_harness.numsim.transpiler.frontend import analyze
from tvm.script import tirx as T


@T.prim_func
def dps_f32_protocol_ops(
    output_f32: T.Buffer((32, 4), "float32"),
    output_f32x2: T.Buffer((32, 4), "uint64"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    tiny: T.let = T.cuda.uint_as_float(T.uint32(1))
    increment: T.let = T.float32(2**-25)
    nan: T.let = T.cuda.uint_as_float(T.uint32(0x7FC00001))

    T.ptx.add.rn.f32(output_f32[lane, 0], T.float32(1), increment)
    T.ptx.sub.rn.f32(output_f32[lane, 1], T.float32(-1), increment)
    T.ptx.mul.rn.f32(output_f32[lane, 2], tiny, T.float32(1))
    T.ptx.fma.rn.f32(output_f32[lane, 3], nan, T.float32(1), T.float32(0))

    packed_lhs: T.let = T.cuda.make_float2(T.float32(1), tiny)
    packed_rhs: T.let = T.cuda.make_float2(increment, T.float32(1))
    packed_addend: T.let = T.cuda.make_float2(tiny, tiny)
    T.ptx.add.rn.f32x2(output_f32x2[lane, 0], packed_lhs, packed_rhs)
    T.ptx.sub.rn.f32x2(output_f32x2[lane, 1], packed_lhs, packed_rhs)
    T.ptx.mul.rn.f32x2(output_f32x2[lane, 2], packed_lhs, packed_rhs)
    T.ptx.fma.rn.f32x2(
        output_f32x2[lane, 3],
        packed_lhs,
        packed_rhs,
        packed_addend,
    )


@T.prim_func
def dps_f64_protocol_ops(output: T.Buffer((32, 4), "float64")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    value: T.let = T.cast(lane, "float64")

    T.ptx.add.rn.f64(output[lane, 0], value, T.float64(2))
    T.ptx.sub.rn.f64(output[lane, 1], value, T.float64(2))
    T.ptx.mul.rn.f64(output[lane, 2], value, T.float64(2))
    T.ptx.fma.rn.f64(output[lane, 3], value, T.float64(2), T.float64(1))


_KERNELS = (
    pointer_conversions_and_descriptor,
    dps_f32_protocol_ops,
    dps_f64_protocol_ops,
    ptx_predicate_data_path,
    ptx_selp_b16_bits,
    ptx_half_abs_and_setp,
    ptx_typed_move_and_b16_pack,
    mma_fragment_fill_and_store,
    raw_dense_f16_f32_k16,
    ptx_mma_legacy_f16_m16n8k16,
    raw_sparse_f16_f32_k32,
    tcgen05_cp_warpx4,
    tcgen05_bf16_mma,
    tcgen05_block_scaled_mxf4,
    raw_tma_roundtrip,
    raw_tma_prefetch,
    raw_tma_gather4_bar_address,
    raw_tma_reduce_add,
    raw_bulk_prefetch,
)


_KERNEL_BY_OP = {
    "tirx.cuda.float22half2": "pointer_conversions_and_descriptor",
    "tirx.cuda.float8tohalf8": "pointer_conversions_and_descriptor",
    "tirx.cuda.half8tofloat8": "pointer_conversions_and_descriptor",
    "tirx.cuda.runtime_instr_desc": "pointer_conversions_and_descriptor",
    "tirx.mma_fill": "mma_fragment_fill_and_store",
    "tirx.mma_fill_legacy": "mma_fragment_fill_and_store",
    "tirx.mma_store": "mma_fragment_fill_and_store",
    "tirx.mma_store_legacy": "mma_fragment_fill_and_store",
    "tirx.ptx.abs_half": "ptx_half_abs_and_setp",
    "tirx.ptx.add": "dps_f32_protocol_ops",
    "tirx.ptx.and": "ptx_predicate_data_path",
    "tirx.ptx.cp_async_bulk_tensor_g2s_cluster": "raw_tma_roundtrip",
    "tirx.ptx.cp_async_bulk_tensor_g2s_cta": "raw_tma_gather4_bar_address",
    "tirx.ptx.cp_async_bulk_tensor_prefetch": "raw_tma_prefetch",
    "tirx.ptx.cp_async_bulk_prefetch": "raw_bulk_prefetch",
    "tirx.ptx.cp_async_bulk_tensor_s2g": "raw_tma_roundtrip",
    "tirx.ptx.cp_reduce_async_bulk_tensor": "raw_tma_reduce_add",
    "tirx.ptx.fma": "dps_f32_protocol_ops",
    "tirx.ptx.mma": "raw_dense_f16_f32_k16",
    "tirx.ptx.mma_sp_pair": "raw_sparse_f16_f32_k32",
    "tirx.ptx.mov_pack_b16x2": "ptx_typed_move_and_b16_pack",
    "tirx.ptx_legacy.mma": "ptx_mma_legacy_f16_m16n8k16",
    "tirx.ptx.mul": "dps_f32_protocol_ops",
    "tirx.ptx.prefetch": "raw_tma_roundtrip",
    "tirx.ptx.selp": "ptx_selp_b16_bits",
    "tirx.ptx.setp": "ptx_predicate_data_path",
    "tirx.ptx.setp_half": "ptx_half_abs_and_setp",
    "tirx.ptx.sub": "dps_f32_protocol_ops",
    "tirx.ptx.tcgen05_cp": "tcgen05_cp_warpx4",
    "tirx.cuda.tcgen05_encode_instr_descriptor": "tcgen05_bf16_mma",
    "tirx.cuda.tcgen05_encode_instr_descriptor_block_scaled": "tcgen05_block_scaled_mxf4",
    "tirx.cuda.tcgen05_encode_matrix_descriptor": "tcgen05_cp_warpx4",
    "tirx.ptx.tcgen05_mma_ss": "tcgen05_bf16_mma",
    "tirx.ptx.tcgen05_mma_block_scale_ss": "tcgen05_block_scaled_mxf4",
}


def _arguments_by_kernel() -> dict[str, dict[str, Any]]:
    tma_roundtrip_source = np.arange(12, dtype=np.float32).reshape(3, 4)
    tma_roundtrip_output = np.zeros_like(tma_roundtrip_source)
    gather_source = np.arange(16, dtype=np.float32).reshape(4, 4)
    reduce_output = np.zeros(4, dtype=np.float32)
    return {
        "pointer_conversions_and_descriptor": {
            "source": np.arange(32 * 8, dtype=np.float32).reshape(32, 8),
            "half": np.zeros((32, 8), dtype=np.float16),
            "roundtrip": np.zeros((32, 8), dtype=np.float32),
            "descriptor": np.zeros(32, dtype=np.uint32),
        },
        "dps_f32_protocol_ops": {
            "output_f32": np.zeros((32, 4), dtype=np.float32),
            "output_f32x2": np.zeros((32, 4), dtype=np.uint64),
        },
        "dps_f64_protocol_ops": {"output": np.zeros((32, 4), dtype=np.float64)},
        "ptx_predicate_data_path": {
            "lhs_f32": np.arange(32, dtype=np.float32) - np.float32(16),
            "lhs_u32": np.arange(32, dtype=np.uint32) % np.uint32(3),
            "selected_u32": np.zeros(32, dtype=np.uint32),
            "selected_f32": np.zeros(32, dtype=np.float32),
        },
        "ptx_selp_b16_bits": {
            "predicate": np.arange(32, dtype=np.uint32) & np.uint32(1),
            "on_true": np.arange(32, dtype=np.uint16) ^ np.uint16(0x8000),
            "on_false": np.arange(32, dtype=np.uint16) ^ np.uint16(0x7C00),
            "selected": np.zeros(32, dtype=np.uint16),
        },
        "ptx_half_abs_and_setp": {
            "packed": np.arange(32, dtype=np.uint32) ^ np.uint32(0x80008000),
            "lhs": np.arange(32, dtype=np.float16).view(np.uint16),
            "rhs": (np.arange(32, dtype=np.float16) - np.float16(1)).view(np.uint16),
            "absolute": np.zeros(32, dtype=np.uint32),
            "greater": np.zeros(32, dtype=np.uint32),
        },
        "ptx_typed_move_and_b16_pack": {
            "low": np.arange(32, dtype=np.uint16),
            "high": np.uint16(0xFFFF) - np.arange(32, dtype=np.uint16),
            "source_i32": np.arange(32, dtype=np.int32) - np.int32(16),
            "packed": np.zeros(32, dtype=np.uint32),
            "unpacked_low": np.zeros(32, dtype=np.uint16),
            "unpacked_high": np.zeros(32, dtype=np.uint16),
            "moved_i32": np.zeros(32, dtype=np.int32),
        },
        "mma_fragment_fill_and_store": {
            "filled": np.zeros((2, 32, 8), dtype=np.float32),
            "stored": np.zeros((2, 16, 16), dtype=np.float32),
        },
        "raw_dense_f16_f32_k16": dict(make_dense_f16_case()),
        "ptx_mma_legacy_f16_m16n8k16": {
            "a": np.zeros((16, 16), dtype=np.float16),
            "b": np.zeros((16, 8), dtype=np.float16),
            "c": np.zeros((16, 8), dtype=np.float32),
            "output": np.zeros((16, 8), dtype=np.float32),
        },
        "raw_sparse_f16_f32_k32": dict(make_sparse_f16_case()),
        "tcgen05_cp_warpx4": {
            "source": np.arange(32 * 4, dtype=np.uint32).reshape(32, 4),
            "output": np.zeros((4, 32, 4), dtype=np.uint32),
        },
        "tcgen05_bf16_mma": {"output": np.zeros((4, 32, 4), dtype=np.float32)},
        "tcgen05_block_scaled_mxf4": {"output": np.zeros((4, 32, 16), dtype=np.float32)},
        "raw_tma_roundtrip": {
            "input_map": _tensor_map(
                tma_roundtrip_source,
                global_shape=(4, 3),
                global_strides=(16,),
                box_shape=(4, 3),
            ),
            "output_map": _tensor_map(
                tma_roundtrip_output,
                global_shape=(4, 3),
                global_strides=(16,),
                box_shape=(4, 3),
            ),
        },
        "raw_tma_prefetch": {
            "input_map": _tensor_map(
                gather_source,
                global_shape=(4, 4),
                global_strides=(16,),
                box_shape=(4, 1),
            ),
            "output": np.zeros(1, dtype=np.int32),
        },
        "raw_tma_gather4_bar_address": {
            "input_map": _tensor_map(
                gather_source,
                global_shape=(4, 4),
                global_strides=(16,),
                box_shape=(4, 1),
            ),
            "output": np.zeros((4, 4), dtype=np.float32),
        },
        "raw_tma_reduce_add": {
            "source": np.arange(4, dtype=np.float32),
            "output_map": _tensor_map(
                reduce_output,
                global_shape=(4,),
                global_strides=(),
                box_shape=(4,),
            ),
        },
        "raw_bulk_prefetch": {
            "source": np.arange(64, dtype=np.uint8) ^ np.uint8(0xA5),
            "num_bytes": np.uint32(32),
            "output": np.zeros(32, dtype=np.uint8),
        },
    }


def _assert_payload_runtime(module, arguments, kernel_name):
    inputs = {
        f"k{index}:{name}": value
        for index, kernel in enumerate(module.spec.kernels)
        for name, value in arguments[kernel.name].items()
    }
    phase_index = next(
        i for i, kernel in enumerate(module.spec.kernels) if kernel.name == kernel_name
    )
    result = (
        numsim.Engine(
            max_workers=1,
            native_loop_iteration_budget=10_000,
            native_loop_reschedule_quantum=16,
        )
        .run_synccheck_phase(
            module,
            inputs,
            phase_index=phase_index,
            coverage_bounds=numsim.CoverageBounds(0, 0),
            resource_limits=numsim.ResourceLimits(
                max_schedules=100,
                max_backtrack_nodes=100_000,
                max_events_per_run=100_000,
                max_total_events=1_000_000,
                max_loop_steps=1_000_000,
                max_wall_time_ms=30_000,
                max_diagnostic_bytes=1_000_000,
            ),
            max_polls=100_000,
            max_transitions=100_000,
        )
        .to_dict()
    )
    assert result["phase"]["name"] == kernel_name
    assert result["execution_error"] is None
    assert result["findings"] == []
    assert result["verdict"] == "clean"
    assert result["incomplete"] == []
    assert result["coverage"]["eligible_for_clean"] is True
    assert result["search"]["algorithm"] == "fixed_sync_state"
    assert result["stats"]["task_count"] > 0
    assert result["stats"]["completed_task_count"] == result["stats"]["task_count"]


@pytest.fixture(scope="module")
def native_payload_artifact(tmp_path_factory):
    spec = analyze(_KERNELS)
    module = numsim.transpile(
        _KERNELS, cache_dir=tmp_path_factory.mktemp("synccheck-payload"), _analysis_capable=True
    )
    assert module.spec.to_manifest(include_source_spans=False) == spec.to_manifest(include_source_spans=False)
    return module, spec


@pytest.mark.parametrize(
    "kernel_name",
    [
        getattr(kernel, "__name__", None) or str(kernel.attrs["global_symbol"])
        for kernel in _KERNELS
    ],
)
def test_payload_runtime(native_payload_artifact, kernel_name):
    module, spec = native_payload_artifact
    kernel = next(kernel for kernel in spec.kernels if kernel.name == kernel_name)
    observed_ops = call_op_names(kernel)
    expected_ops = {op for op, owner in _KERNEL_BY_OP.items() if owner == kernel_name}
    if kernel_name == "dps_f64_protocol_ops":
        expected_ops = {"tirx.ptx.add", "tirx.ptx.sub", "tirx.ptx.mul", "tirx.ptx.fma"}
    assert expected_ops
    assert expected_ops <= observed_ops, (kernel_name, expected_ops - observed_ops)
    _assert_payload_runtime(module, _arguments_by_kernel(), kernel_name)
