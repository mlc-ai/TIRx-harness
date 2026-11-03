"""Synthetic sequential-launch kernels for multi-kernel artifact tests."""

from tvm_tirx.script import tirx as T


@T.function
def write_intermediate(intermediate: T.Tensor((32,), "float32")):
    T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=32))
    lane = T.cuda.lane_id()
    intermediate[lane] = T.cast(lane + 1, "float32")


@T.function
def consume_intermediate(
    intermediate: T.Tensor((32,), "float32"), output: T.Tensor((2, 32), "float32")
):
    T.device_entry(launch=T.cuda.LaunchConfig(grid=2, block=32))
    cta = T.cuda.block_idx("x")
    lane = T.cuda.lane_id()
    output[cta, lane] = intermediate[lane] * T.float32(2) + T.cast(cta, "float32")


@T.function
def ambiguous_alias_first(shared: T.Tensor((32,), "float32")):
    T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=32))
    lane = T.cuda.lane_id()
    shared[lane] = T.float32(1)


@T.function
def ambiguous_alias_second(shared: T.Tensor((32,), "float32")):
    T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=32))
    lane = T.cuda.lane_id()
    shared[lane] = T.float32(2)


@T.function
def same_names_first(
    input: T.Tensor((32,), "float32"),
    intermediate: T.Tensor((32,), "float32"),
    output: T.Tensor((32,), "float32"),
    scale: T.int32,
):
    T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=32))
    lane = T.cuda.lane_id()
    intermediate[lane] = input[lane] * T.cast(scale, "float32")
    output[lane] = input[lane] + T.float32(1)


@T.function
def same_names_second(
    input: T.Tensor((32,), "float32"),
    intermediate: T.Tensor((32,), "float32"),
    output: T.Tensor((32,), "float32"),
    scale: T.float32,
):
    T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=32))
    lane = T.cuda.lane_id()
    output[lane] = input[lane] + intermediate[lane] * scale


@T.function
def typed_pointer_first(pointer: T.handle("uint32"), output: T.Tensor((32,), "uint32")):
    T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=32))
    lane = T.cuda.lane_id()
    address = T.ptr_byte_offset(pointer, T.cast(lane * 4, "uint32"), ty=T.Ptr("uint32"))
    T.ptx.ld.global_.u32(output[lane], address)


@T.function
def typed_pointer_second(pointer: T.handle("uint32"), output: T.Tensor((32,), "uint32")):
    T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=32))
    lane = T.cuda.lane_id()
    address = T.ptr_byte_offset(pointer, T.cast(lane * 4, "uint32"), ty=T.Ptr("uint32"))
    loaded = T.local_scalar("uint32")
    T.ptx.ld.global_.u32(loaded, address)
    output[lane] = loaded + T.uint32(1)
