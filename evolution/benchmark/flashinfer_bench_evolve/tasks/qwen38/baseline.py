"""Upstream input, cache, correctness and timing helpers; callable-based Runner.

Adapted from mlc-ai/qwen38-inference at
05e8ef0db302a2c005994b79d9a5e78b96e2d88b (Apache-2.0).
"""
import math
import time

import torch


def inputs(case, seed):
    generator = torch.Generator().manual_seed(seed + sum(map(ord, case.name)))
    rows = [torch.randint(100, 20000, (p+n,), generator=generator).tolist()
            for p,n in zip(case.previous, case.lengths)]
    return rows, [row[p:] for row,p in zip(rows,case.previous)]


def new_cache(model, case):
    capacities = [math.ceil((p+n)/64)*64 for p,n in zip(case.previous,case.lengths)]
    offsets = [0]
    for size in capacities:
        offsets.append(offsets[-1]+size)
    kv,conv,recurrent = {},{},{}
    for i,kind in enumerate(model.layer_types):
        if kind == "full_attention":
            kv[i] = tuple(torch.zeros((offsets[-1]+64,4,256),dtype=model.dtype,device=model.device) for _ in range(2))
        else:
            conv[i] = torch.zeros((case.batch,10240,3),dtype=model.dtype,device=model.device)
            recurrent[i] = torch.zeros((case.batch,48,128,128),dtype=torch.float32,device=model.device)
    from .model import HybridCache

    cache = HybridCache(id(model),[0]*case.batch,max(capacities),131072,kv,conv,recurrent)
    cache.slot_offsets = offsets[:-1]
    cache.slot_capacities = capacities
    cache.verify_conv, cache.verify_conv_storage, cache.verify_recurrent = {}, {}, {}
    if case.all_logits:
        for i in conv:
            physical = torch.zeros((case.batch, 10240, 6), dtype=model.dtype, device=model.device)
            cache.verify_conv_storage[i] = physical
            cache.verify_conv[i] = physical.as_strided((case.batch, 4, 10240, 3),
                                                      (physical.stride(0), 1, 6, 1))
            cache.verify_recurrent[i] = torch.zeros((case.batch, 4, 48, 128, 128),
                                                    dtype=torch.float32, device=model.device)
    return cache


def prime(model, cache, case, rows):
    """Compute real prefix states in bounded chunks before any timed call."""
    while cache.lengths != case.previous:
        slots, chunks, budget = [],[],32768
        for slot,target in enumerate(case.previous):
            start = cache.lengths[slot]
            count = min(target-start,8192,budget)
            if count:
                slots.append(slot)
                chunks.append(rows[slot][start:start+count])
                budget -= count
            if not budget:
                break
        model.forward_step(chunks,cache,request_indices=slots)


def state_tensors(cache):
    return list(cache.conv.values()) + list(cache.recurrent.values())


def all_tensors(cache):
    return ([x for pair in cache.kv.values() for x in pair]+state_tensors(cache)
            +list(cache.verify_conv_storage.values())+list(cache.verify_recurrent.values()))


def snapshot(tensor):
    """One independent CPU copy, pinned for CUDA transfers."""
    output = torch.empty_like(tensor, device="cpu", pin_memory=tensor.is_cuda)
    output.copy_(tensor)
    return output


def prefix_ranges(lengths):
    """Contiguous request ranges whose initial states contain a real prefix."""
    start = None
    for index, length in enumerate([*lengths, 0]):
        if length and start is None:
            start = index
        elif not length and start is not None:
            yield start, index
            start = None


def save_state(cache):
    """Snapshot populated request ranges; omitted requests restore to zero."""
    tensors = state_tensors(cache)
    return [(start, end, [snapshot(t[start:end]) for t in tensors])
            for start, end in prefix_ranges(cache.lengths)]


def restore(cache, saved, prepared):
    for index, dst in enumerate(state_tensors(cache)):
        cursor = 0
        for start, end, tensors in saved:
            dst[cursor:start].zero_()
            dst[start:end].copy_(tensors[index])
            cursor = end
        dst[cursor:].zero_()
    for pair in cache.kv.values():
        for tensor in pair:
            tensor.index_fill_(0,prepared["locations"],0)
    for tensor in list(cache.verify_conv_storage.values())+list(cache.verify_recurrent.values()):
        tensor.zero_()


def check(actual, expected, label):
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        raise AssertionError(f"{label}: shape or dtype mismatch")
    a,e = actual.reshape(-1),expected.reshape(-1)
    worst = 0.0
    for start in range(0,a.numel(),4_194_304):
        x = a[start:start+4_194_304]
        y = e[start:start+4_194_304].to(x.device)
        torch.testing.assert_close(x,y,rtol=1e-3,atol=1e-3,equal_nan=False,msg=lambda msg: label+"\n"+msg)
        if not bool(torch.isfinite(x).all()):
            raise AssertionError(f"{label}: nonfinite output")
        worst = max(worst,float((x-y).abs().max()))
    return worst


class Runner:
    def __init__(self, forward, graph, reset):
        self.forward = forward
        self.graph = None
        if graph:
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    reset()
                    forward()
            torch.cuda.current_stream().wait_stream(stream)
            reset()
            torch.cuda.synchronize()
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                self.output = forward()

    def __call__(self):
        if self.graph is None:
            return self.forward()
        self.graph.replay()
        return self.output


def timed(fn):
    torch.cuda.synchronize()
    begin,end = torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
    start = time.perf_counter()
    begin.record()
    output = fn()
    end.record()
    end.synchronize()
    return output,{"wall_ms":(time.perf_counter()-start)*1000,"gpu_ms":begin.elapsed_time(end)}
