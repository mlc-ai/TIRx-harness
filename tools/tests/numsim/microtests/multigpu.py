"""Run one multimem case on N GPUs over torch symmetric memory and in NumSim."""

from __future__ import annotations

import os
from pathlib import Path
import socket
import sys
import tempfile

import numpy as np

from tirx_harness import numsim


class _CudaArray:
    """A raw device pointer, viewable by torch, for the multicast address."""

    def __init__(self, pointer: int, shape: tuple[int, ...], dtype: np.dtype):
        self.__cuda_array_interface__ = {
            "shape": shape,
            "typestr": np.dtype(dtype).str,
            "data": (int(pointer), False),
            "version": 3,
        }


def _worker(rank: int, world: int, port: int, case_name: str, out_dir: str, path: list[str]):
    sys.path[:] = path
    import torch
    import torch.distributed as dist
    import torch.distributed._symmetric_memory as symm_mem
    import tvm

    from tests.numsim.microtests.cases.multimem import CASES_BY_NAME

    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group("nccl", rank=rank, world_size=world, device_id=device)
    try:
        case = CASES_BY_NAME[case_name]
        arguments = case.rank_arguments(rank, world)
        target = tvm.target.Target({"kind": "cuda", "arch": "sm_100a"})
        with target:
            executable = tvm.compile(
                tvm.IRModule({"main": case.prim_func()}), target=target, tir_pipeline="tirx"
            )
        replica = arguments["data"]
        data = symm_mem.empty(replica.shape, dtype=torch.from_numpy(replica).dtype, device=device)
        handle = symm_mem.rendezvous(data, dist.group.WORLD.group_name)
        data.copy_(torch.from_numpy(replica))
        mc = torch.as_tensor(_CudaArray(handle.multicast_ptr, replica.shape, replica.dtype),
                             device=device)
        src = torch.from_numpy(arguments["src"]).to(device)
        out = torch.from_numpy(arguments["out"]).to(device)
        torch.cuda.synchronize()
        handle.barrier()
        executable(mc, data, src, out, int(arguments["rank"]))
        torch.cuda.synchronize()
        handle.barrier()
        for name, tensor in (("data", data), ("out", out)):
            np.save(os.path.join(out_dir, f"{name}{rank}.npy"), tensor.cpu().numpy())
    finally:
        dist.destroy_process_group()


def run_gpu_case(case_name: str, world: int, *, attempts: int = 3) -> list[dict[str, np.ndarray]]:
    import torch.multiprocessing as mp

    for attempt in range(attempts):
        # The probed port can be taken again before the rendezvous binds it.
        with socket.create_server(("127.0.0.1", 0)) as probe:
            port = probe.getsockname()[1]
        with tempfile.TemporaryDirectory() as out_dir:
            try:
                mp.spawn(_worker, args=(world, port, case_name, out_dir, list(sys.path)),
                         nprocs=world)
            except mp.ProcessRaisedException as error:
                if "EADDRINUSE" in str(error) and attempt + 1 < attempts:
                    continue
                raise
            return [
                {name: np.load(os.path.join(out_dir, f"{name}{rank}.npy"))
                 for name in ("data", "out")}
                for rank in range(world)
            ]
    raise AssertionError("unreachable")


def _resolve(spec: str):
    module, name = spec.split(":")
    return getattr(__import__(module, fromlist=[name]), name)


def _kernel_worker(rank: int, world: int, port: int, kernel: tuple, inputs: tuple,
                   outputs: tuple[str, ...], out_dir: str, path: list[str]):
    sys.path[:] = path
    import torch
    import torch.distributed as dist
    import torch.distributed._symmetric_memory as symm_mem
    import tvm

    from tests.numsim.microtests.harness import PairedTensorMap, _torch_tensor_map

    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group("nccl", rank=rank, world_size=world, device_id=device)
    try:
        func = _resolve(kernel[0])
        if kernel[1] is not None:
            func = func(*kernel[1])
        bindings = _resolve(inputs[0])(*inputs[1], **inputs[2])[rank]
        windows = {name: value for name, value in bindings.items()
                   if isinstance(value, numsim.MulticastWindow)}
        symmetric = {}
        for window_name, window in windows.items():
            replica = next(name for name, value in bindings.items()
                           if value is window.replicas[rank])
            symmetric[replica] = window_name
        tensors = {}
        for name, value in bindings.items():
            if isinstance(value, numsim.MulticastWindow):
                continue
            if name in symmetric:
                tensor = symm_mem.empty(value.shape, dtype=torch.from_numpy(value).dtype,
                                        device=device)
                handle = symm_mem.rendezvous(tensor, dist.group.WORLD.group_name)
                tensor.copy_(torch.from_numpy(value))
                tensors[symmetric[name]] = torch.as_tensor(
                    _CudaArray(handle.multicast_ptr, value.shape, value.dtype), device=device)
                tensors[name] = tensor
            elif isinstance(value, np.ndarray):
                tensors[name] = torch.from_numpy(value).to(device)
            elif not isinstance(value, PairedTensorMap):
                tensors[name] = int(value)
        tensor_maps = []
        for name, value in bindings.items():
            if isinstance(value, PairedTensorMap):
                base = next((tensors[other] for other, array in bindings.items()
                             if array is value.array), None)
                tensor_maps.append(_torch_tensor_map(value, base=base))
                tensors[name] = tensor_maps[-1].descriptor
        target = tvm.target.Target({"kind": "cuda", "arch": "sm_100a"})
        with target:
            executable = tvm.compile(tvm.IRModule({"main": func}), target=target,
                                     tir_pipeline="tirx")
        torch.cuda.synchronize()
        dist.barrier()
        executable(*(tensors[param.name] for param in func.params))
        torch.cuda.synchronize()
        dist.barrier()
        for name in outputs:
            np.save(os.path.join(out_dir, f"{name}{rank}.npy"), tensors[name].cpu().numpy())
    finally:
        dist.destroy_process_group()


def run_gpu_kernel(kernel: tuple[str, tuple], inputs: tuple[str, tuple, dict], world: int,
                   outputs: tuple[str, ...], *, attempts: int = 3) -> list[dict[str, np.ndarray]]:
    """Launch a TIRx kernel once on each of `world` GPUs.

    `kernel` is ``("module:factory", args)`` building the prim_func, or
    ``("module:prim_func", None)`` naming it, and
    `inputs` is ``("module:factory", args, kwargs)`` building the per-rank
    NumSim bindings; every array a `MulticastWindow` covers is allocated in
    torch symmetric memory and the window binds its multicast address. A
    `PairedTensorMap` over a bound array encodes against that array's device copy.
    """
    import torch.multiprocessing as mp

    for attempt in range(attempts):
        with socket.create_server(("127.0.0.1", 0)) as probe:
            port = probe.getsockname()[1]
        with tempfile.TemporaryDirectory() as out_dir:
            try:
                mp.spawn(_kernel_worker,
                         args=(world, port, kernel, inputs, outputs, out_dir, list(sys.path)),
                         nprocs=world)
            except mp.ProcessRaisedException as error:
                if "EADDRINUSE" in str(error) and attempt + 1 < attempts:
                    continue
                raise
            return [
                {name: np.load(os.path.join(out_dir, f"{name}{rank}.npy")) for name in outputs}
                for rank in range(world)
            ]
    raise AssertionError("unreachable")


def run_numsim_case(case, world: int, *, cache_dir: str | Path | None = None):
    ranks = [case.rank_arguments(rank, world) for rank in range(world)]
    window = numsim.MulticastWindow([arguments["data"] for arguments in ranks])
    inputs = [{**arguments, "mc": window} for arguments in ranks]
    module = numsim.transpile(case.prim_func(), cache_dir=cache_dir)
    result = numsim.Engine().run(module, inputs, outputs=["out", "data"])
    return [
        {name: np.asarray(result.outputs[numsim.rank_binding_name(name, rank)])
         for name in ("data", "out")}
        for rank in range(world)
    ]


def mismatches(case, gpu, sim) -> list[str]:
    """Per-element differences beyond `case.tolerance_ulps`, with every rank's
    input for the first few."""

    world = len(gpu)
    inputs = [case.rank_arguments(rank, world) for rank in range(world)]
    lane = case.lane_dtype
    operand = "data" if case.kind == "ld_reduce" else "src"
    report = []
    for name in case.outputs:
        for rank in range(world):
            g = np.ascontiguousarray(gpu[rank][name])
            s = np.ascontiguousarray(sim[rank][name])
            assert g.shape == s.shape and g.dtype == s.dtype, (name, g.dtype, s.dtype)
            g, s = g.view(lane), s.view(lane)
            differ = g != s
            if case.tolerance_ulps:
                differ &= np.abs(_ordered(g) - _ordered(s)) > case.tolerance_ulps
            bad = np.flatnonzero(differ & ~(_is_nan(g, case) & _is_nan(s, case)))
            for index in bad[:4]:
                contributions = [hex(int(_lanes(i[operand], lane)[index])) for i in inputs]
                report.append(
                    f"{name}@rank{rank}[{index}]: gpu={hex(int(g[index]))} "
                    f"numsim={hex(int(s[index]))} inputs={contributions}"
                    + (f" initial={hex(int(_lanes(inputs[rank]['data'], lane)[index]))}"
                       if case.kind == "red" else "")
                )
            if len(bad) > 4:
                report.append(f"{name}@rank{rank}: {len(bad)} mismatching elements")
    return report


def _lanes(array, lane: np.dtype) -> np.ndarray:
    return np.ascontiguousarray(array).view(lane)


def _ordered(bits: np.ndarray) -> np.ndarray:
    """Float bit patterns mapped to integers that step by one per ulp."""
    values = bits.astype(np.int64)
    sign = 1 << (bits.dtype.itemsize * 8 - 1)
    return np.where(values >= sign, -(values - sign), values)


def _is_nan(bits: np.ndarray, case) -> np.ndarray:
    element = case.form.rsplit(".", 1)[1].replace("x2", "")
    exponent, fraction = {
        "f16": (0x7C00, 0x03FF), "bf16": (0x7F80, 0x007F),
        "f32": (0x7F800000, 0x007FFFFF), "f64": (0x7FF0000000000000, 0x000FFFFFFFFFFFFF),
    }.get(element, (None, None))
    if exponent is None:
        return np.zeros(bits.shape, bool)
    return ((bits & exponent) == exponent) & ((bits & fraction) != 0)
