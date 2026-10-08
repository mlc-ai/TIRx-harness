"""Capture production dispatch on CPU, preserving its kernels and argument aliases."""

from contextlib import ExitStack
from importlib.util import module_from_spec, spec_from_file_location
import sys
from types import SimpleNamespace
from unittest.mock import patch

import ctypes
import numpy as np
import torch
import tvm

from tests.numsim.support._tirx_kernels import load_tirx_kernel
from tirx_harness.numsim.cases import ComparisonSpec, NumSimCase, TensorMap


def _numpy(value):
    if not isinstance(value, torch.Tensor):
        return value
    if value.dtype == torch.bfloat16:
        return value.view(torch.uint16).numpy()
    if value.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        return value.view(torch.uint8).numpy()
    return value.numpy()


class _CpuTorch:
    def __getattr__(self, name):
        return getattr(torch, name)

    @staticmethod
    def device(*args, **kwargs):
        return torch.device("cpu")


def _prepare(name, config, launch=None):
    canonical = load_tirx_kernel(name)
    # Dispatch factories also hide caches in closures. Load the canonical
    # source in a private module so capture executables never escape into a
    # later GPU launch through those caches.
    private_name = canonical.__package__ + "._numsim_" + name
    spec = spec_from_file_location(private_name, canonical.__file__)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    with patch.dict(sys.modules, {private_name: module}):
        spec.loader.exec_module(module)
    calls, tensors, descriptors = [], {}, {}
    data_ptr = torch.Tensor.data_ptr
    get_global_func = tvm.get_global_func

    def pointer(tensor):
        address = data_ptr(tensor)
        tensors[address] = tensor
        return address

    def address(value):
        return value if isinstance(value, int) else ctypes.cast(value, ctypes.c_void_p).value

    def encode(dest, dtype, rank, base, *fields):
        dims = fields[:rank]
        strides = fields[rank : 2 * rank - 1]
        box = fields[2 * rank - 1 : 3 * rank - 1]
        element = fields[3 * rank - 1 : 4 * rank - 1]
        interleave, swizzle, _l2, fill = fields[4 * rank - 1 :]
        assert interleave == 0 and fill == 0
        descriptors[address(dest)] = TensorMap(
            base=_numpy(tensors[address(base)]),
            dtype=str(dtype),
            global_shape=tuple(dims),
            global_strides=tuple(strides),
            box_shape=tuple(box),
            element_strides=tuple(element),
            swizzle={0: None, 1: "32B", 2: "64B", 3: "128B"}[swizzle],
        ).numpy()

    class Executable:
        def __init__(self, func):
            if isinstance(func, tvm.IRModule):
                funcs = list(func.functions.values())
                assert len(funcs) == 1
                func = funcs[0]
            self.func = func

        def __call__(self, *args):
            calls.append((self.func, args))

        def jit(self):
            return self

        @property
        def main(self):
            return self

        def __getitem__(self, key):
            return self

    def compile_kernel(func, *args, **kwargs):
        return Executable(func)

    with ExitStack() as stack:
        stack.enter_context(patch.object(module, "torch", _CpuTorch()))
        stack.enter_context(patch.object(torch.Tensor, "data_ptr", pointer))
        stack.enter_context(
            patch.object(
                torch.cuda,
                "get_device_properties",
                return_value=SimpleNamespace(multi_processor_count=1, major=10, minor=0),
            )
        )
        stack.enter_context(patch.object(torch.cuda, "synchronize"))
        stack.enter_context(patch.object(torch.cuda, "current_device", return_value=0))
        stack.enter_context(patch("tirx_kernels.runner.hardware_num_sms", return_value=1))
        stack.enter_context(patch("tirx_kernels.runner.compile_kernel", compile_kernel))
        stack.enter_context(patch.object(tvm, "compile", compile_kernel))
        stack.enter_context(
            patch.object(
                tvm,
                "get_global_func",
                lambda name, *a, **k: encode
                if name == "runtime.cuTensorMapEncodeTiled"
                else get_global_func(name, *a, **k),
            )
        )
        case = module.prepare_data(**config)
        expected = module._reference_output(case)
        run = launch(module, case) if launch else module._launch_state(case)
        calls.clear()
        run()

    assert calls, name
    args = {}
    outputs = {}
    references = (
        {"output": expected[0], "final_state": expected[1]}
        if isinstance(expected, tuple)
        else {"output": expected}
    )
    for i, (func, values) in enumerate(calls):
        for param, value in zip(func.params, values, strict=True):
            key = f"k{i}:{param.name}" if len(calls) > 1 else str(param.name)
            if isinstance(value, (ctypes.c_void_p, ctypes.Array)):
                value = descriptors[address(value)]
            elif isinstance(value, int) and value in descriptors:
                value = descriptors[value]
            args[key] = _numpy(value)
            if isinstance(value, torch.Tensor):
                for output in references:
                    if data_ptr(value) == data_ptr(case[output]):
                        outputs[output] = key
    # Tensor-map-only outputs retain the real contiguous output owner.
    from tirx_harness.numsim.bindings import _decode_tensor_maps, _tensor_map_base_array

    for key, value in args.items():
        if isinstance(value, np.ndarray) and _decode_tensor_maps(value):
            base = _tensor_map_base_array(value)
            for output in references:
                if np.shares_memory(base, _numpy(case[output])):
                    outputs[output] = key
    assert outputs.keys() == references.keys(), (name, outputs, args.keys())
    expected_arrays = {}
    for output, reference in references.items():
        binding = args[outputs[output]]
        maps = _decode_tensor_maps(binding)
        values = reference.float().contiguous().numpy()
        if maps:
            descriptor = maps[0]
            # Tensor-map results use descriptor dimension order; apply the same
            # physical view to the independent FP32 oracle without rounding it.
            assert descriptor.dtype == "bfloat16"
            expected_arrays[output] = np.ndarray(
                shape=tuple(reversed(descriptor.global_shape)),
                dtype=np.float32,
                buffer=values,
                strides=tuple(reversed((4, *(stride * 2 for stride in descriptor.global_strides)))),
            ).copy()
        else:
            expected_arrays[output] = values.reshape(binding.shape)
    atol, rtol = (8e-4, 2e-2) if "mla_dsv4" in name else (1e-2, 1e-2)
    return NumSimCase(
        kernel=tuple(func for func, _ in calls) if len(calls) > 1 else calls[0][0],
        args=args,
        outputs=outputs,
        reference=lambda: expected_arrays,
        comparisons={
            key: ComparisonSpec(atol=atol, rtol=rtol, actual_encoding="bfloat16")
            for key in references
        },
    )


def prepare_native_msa_case(*, build_case=_prepare):
    return build_case(
        "msa_prefill_multishape",
        dict(
            batch_size=1,
            seqlen_q=32,
            seqlen_kv=256,
            num_qo_heads=16,
            num_kv_heads=1,
            topk=4,
            kv_layout="flat",
            kv_dtype="bfloat16",
        ),
        lambda module, case: module.setup(case, 32, 1),
    )


def prepare_native_vsa_case(*, build_case=_prepare):
    return build_case(
        "vsa_multishape", dict(seq_len=384, num_heads=1, topk=2, block_size=128)
    )


def prepare_native_kda_decode_case(*, build_case=_prepare):
    return build_case(
        "kda_decode_multishape",
        dict(num_tokens=1, num_seqs=1, num_v_heads=16, lower_bound_gate=False),
    )


def prepare_native_msa_decode_case(*, build_case=_prepare):
    return build_case(
        "msa_decode_multishape",
        dict(
            batch_size=1,
            seqlen_q=1,
            seqlen_kv=256,
            num_qo_heads=16,
            num_kv_heads=1,
            topk=2,
            kv_layout="flat",
            kv_dtype="bfloat16",
        ),
    )


def prepare_native_mla_case(*, build_case=_prepare):
    return build_case(
        "mla_dsv4_multishape",
        dict(label="guard_seqlens_h64_swa128_swa128_bf16_hnd", num_seqs=1, max_q_len=1),
    )
