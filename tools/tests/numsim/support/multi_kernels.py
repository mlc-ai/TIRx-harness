"""Synthetic sequential-launch kernels for multi-kernel artifact tests."""

from tvm.script import tirx as T


@T.prim_func
def write_intermediate(intermediate: T.Buffer((32,), "float32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    intermediate[lane] = T.cast(lane + 1, "float32")


@T.prim_func
def consume_intermediate(
    intermediate: T.Buffer((32,), "float32"), output: T.Buffer((2, 32), "float32")
):
    T.device_entry()
    cta = T.cta_id([2])
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    output[cta, lane] = intermediate[lane] * T.float32(2) + T.cast(cta, "float32")


@T.prim_func
def ambiguous_alias_first(shared: T.Buffer((32,), "float32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared[lane] = T.float32(1)


@T.prim_func
def ambiguous_alias_second(shared: T.Buffer((32,), "float32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared[lane] = T.float32(2)


@T.prim_func
def same_names_first(
    input: T.Buffer((32,), "float32"),
    intermediate: T.Buffer((32,), "float32"),
    output: T.Buffer((32,), "float32"),
    scale: T.int32,
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    intermediate[lane] = input[lane] * T.cast(scale, "float32")
    output[lane] = input[lane] + T.float32(1)


@T.prim_func
def same_names_second(
    input: T.Buffer((32,), "float32"),
    intermediate: T.Buffer((32,), "float32"),
    output: T.Buffer((32,), "float32"),
    scale: T.float32,
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    output[lane] = input[lane] + intermediate[lane] * scale


@T.prim_func
def typed_pointer_first(pointer: T.handle("uint32"), output: T.Buffer((32,), "uint32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    address = T.ptr_byte_offset(pointer, T.cast(lane * 4, "uint32"), "uint32")
    T.ptx.ld.global_.u32(output[lane], address)


@T.prim_func
def typed_pointer_second(pointer: T.handle("uint32"), output: T.Buffer((32,), "uint32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    address = T.ptr_byte_offset(pointer, T.cast(lane * 4, "uint32"), "uint32")
    loaded = T.local_scalar("uint32")
    T.ptx.ld.global_.u32(loaded, address)
    output[lane] = loaded + T.uint32(1)
