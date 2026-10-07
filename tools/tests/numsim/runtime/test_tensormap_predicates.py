"""TensorMap predicates gate updates, publication, and pointer resolution."""

import numpy as np
import tvm
from tvm.script import tirx as T

from tirx_harness import numsim, racecheck, synccheck
from tests.numsim.support.execution import run_checked


# Address-space dispatch and integer/float bit carriers are independent.
TENSOR_MAP_PREDICATE_CASES = (
    ("shared::cta", "uint"), ("global", "uint"), ("", "uint"),
    ("global", "int"), ("shared::cta", "float"),
)


def tensor_map_predicate_case(space, *, fault=None, carrier="uint"):
    shared = space != "global"
    image = "image" if shared else "destination"
    pointer = f"{image}.ptr_to([lane * 128])"
    invalid_pointer = f"{image}.ptr_to([1024])"
    if not space:
        pointer = f'T.reinterpret("handle", T.reinterpret("uint64", {pointer}))'
        invalid_pointer = f'T.reinterpret("handle", T.reinterpret("uint64", {invalid_pointer}))'
    qualifier = f"{space}." if space else ""
    # Every replacement entry available on SM100, using one compatible image.
    # The non-issuing lanes have different values and out-of-bounds pointers.
    fields = (
        ("global_address", "b64", "", 'T.reinterpret("uint64", replacement.ptr_to([lane * 4]))'),
        ("rank", "b32", "", "T.uint32(lane + 1)"),
        ("global_dim", "b32", "0, ", "T.uint32(lane + 2)"),
        ("box_dim", "b32", "0, ", "T.uint32(lane + 4)"),
        ("element_stride", "b32", "1, ", "T.uint32(lane + 1)"),
        ("global_stride", "b64", "0, ", "T.uint64(lane + 32)"),
        ("elemtype", "b32", "", "2"),
        ("interleave_layout", "b32", "", "0"),
        ("swizzle_mode", "b32", "", "0"),
        ("fill_mode", "b32", "", "0"),
    )
    updates = []
    for field, width, ordinal, value in fields:
        if fault == "value" and field == "rank":
            value = "T.uint32(0xffffffff)"
        if carrier != "uint" and not value.isdecimal():
            # Dimension fields have the narrower b32i domain, unlike rank/b64.
            kind = "int" if carrier == "float" and ordinal and width == "b32" else carrier
            value = f'T.reinterpret("{kind}{width[1:]}", {value})'
        updates.append(
            f'    T.ptx["tensormap_replace.tile.{field}.{qualifier}b1024.{width}"]('
            f"{pointer}, {ordinal}{value}, pred=lane == 0 and enabled != 0)"
        )
    updates = "\n".join(updates)
    publish = (
        'T.ptx["tensormap_cp_fenceproxy.global.shared::cta.tensormap::generic.release.gpu.sync.aligned"]('
        "destination.ptr_to([0]), image.ptr_to([0]), "
        f"pred=enabled != 0{' and lane == 0' if fault == 'copy' else ''})"
        if shared
        else "T.ptx.fence.proxy.tensormap__generic.release.gpu("
        f"pred=lane == 0 and enabled {'< 0' if fault == 'release' else '!= 0'})"
    )
    kernel = tvm.script.from_source(
        f"""
@T.prim_func
def kernel(descriptor: T.Buffer((128,), "uint8"), destination: T.Buffer((128,), "uint8"),
           replacement: T.Buffer((16,), "uint32"), output: T.Buffer((4,), "uint32"), enabled: T.int32):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    image = T.alloc_buffer((128,), "uint8", scope="shared", align=128)
    data = T.alloc_buffer((4,), "uint32", scope="shared", align=128)
    barrier = T.alloc_buffer((1,), "uint64", scope="shared")
    for i in T.serial(4):
        {image}[i * 32 + lane] = descriptor[i * 32 + lane]
    T.cuda.cta_sync()
{updates}
    T.cuda.warp_sync()
    T.ptx["tensormap_replace.tile.box_dim.{qualifier}b1024.b32"](
        {invalid_pointer}, 0, T.uint32(0), pred=enabled < 0)
    T.ptx.fence.proxy.tensormap__generic.acquire.gpu(destination.ptr_to([1024]), pred=enabled < 0)
    T.ptx.fence.proxy.tensormap__generic.release.gpu(pred=enabled < 0)
    T.ptx["tensormap_cp_fenceproxy.global.shared::cta.tensormap::generic.release.gpu.sync.aligned"](
        destination.ptr_to([1024]), image.ptr_to([1024]), pred=enabled < 0)
    {publish}
    T.ptx.fence.proxy.tensormap__generic.acquire.gpu(
        destination.ptr_to([lane * 128]), pred=lane == 0 and enabled {"< 0" if fault == "acquire" else "!= 0"})
    if lane == 0:
        T.ptx.mbarrier.init.shared.b64(barrier.ptr_to([0]), 1)
    T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.cta_sync()
    if enabled != 0:
        if lane == 0:
            T.ptx.mbarrier.arrive.expect_tx.shared.b64(barrier.ptr_to([0]), 16)
            T.ptx["cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes"](
                data.ptr_to([0]), destination.ptr_to([0]), 0, 1, barrier.ptr_to([0]))
            T.cuda.mbarrier_wait(barrier.ptr_to([0]), 0)
        T.cuda.cta_sync()
        if lane < 4:
            output[lane] = data[lane]
""",
        {"T": T},
    )
    source = np.arange(1, 9, dtype=np.uint32)
    metadata = dict(
        global_shape=(4, 2), global_strides=(16,), box_shape=(4, 1), element_strides=(1, 1)
    )
    inputs = dict(
        descriptor=numsim.TensorMap(base=source, **metadata).numpy(),
        destination=np.zeros(128, np.uint8),
        replacement=np.arange(41, 57, dtype=np.uint32),
        output=np.full(4, 0xDEADBEEF, np.uint32),
        enabled=1,
    )
    return kernel, inputs, source, metadata


def check_tensor_map_predicates(kernel, inputs):
    # Engine output writeback must not change the next invocation's canaries.
    inputs = {
        **inputs,
        "destination": inputs["destination"].copy(),
        "output": inputs["output"].copy(),
        "replacement": inputs["replacement"].copy(),
    }
    result = run_checked(kernel, inputs, outputs=("output",))
    # Row 1 starts eight u32 elements into the replacement: the new 32-byte
    # stride must be observed, not the old 16-byte stride or a float-to-int zero.
    expected = [49, 50, 0, 0] if inputs["enabled"] else [0xDEADBEEF] * 4
    np.testing.assert_array_equal(result.outputs["output"], expected)
    return result.outputs["output"]


def test_tensor_map_predicate_effects():
    for space, carrier in TENSOR_MAP_PREDICATE_CASES:
        kernel, inputs, _, _ = tensor_map_predicate_case(space, carrier=carrier)
        for enabled in (1, 0):
            inputs["enabled"] = enabled
            check_tensor_map_predicates(kernel, inputs)


def test_tensor_map_predicates_retain_errors():
    for space, fault, message in (
        ("shared::cta", "copy", "32 lanes"),
        ("shared::cta", "acquire", "acquire"),
        ("global", "release", "dirty"),
        ("global", "acquire", "acquire"),
        ("shared::cta", "value", "4294967295"),
    ):
        kernel, inputs, _, _ = tensor_map_predicate_case(space, fault=fault, carrier="float")
        for checker in (synccheck, racecheck):
            result = checker(kernel, inputs)
            assert result.verdict == "error", result.format()
            assert "tensormap" in result.format().lower()
            assert message in result.format().lower(), result.format()
