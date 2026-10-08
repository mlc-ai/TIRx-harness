"""FP32 atomic addition: independent bit boundaries and unchanged memory contracts."""

import numpy as np
import pytest
import tvm
from tvm.script import tirx as T

from tirx_harness import numsim, racecheck, synccheck


ATOMIC_CASES = [
    (kind, width, space)
    for kind in ("atom", "sink", "red")
    for width, space in ((1, "global"), (1, "shared::cta"), (2, "global"), (4, ""))
]


def atomic_kernel(kind, width, space, *, noftz=True, offset=0, race=False):
    mnemonic = "red" if kind == "red" else "atom"
    tokens = [mnemonic, "relaxed", "cta", space, "add", "noftz" if noftz else ""]
    tokens += [f"v{width}" if width > 1 else "", "f32"]
    spelling = ".".join(token for token in tokens if token)
    pointer = "shared" if space.startswith("shared") else "destination"
    arguments = [f"{pointer}.ptr_to([lane * {width} + {offset}])"]
    values = [f"value[lane * {width} + {i}]" for i in range(width)]
    if kind == "atom":
        arguments.insert(0, ", ".join(f"old[{i}]" for i in range(width)))
    arguments.extend(values)
    if width > 1:
        arguments.append("pred=lane % 2 == 0")
        if kind == "atom":
            arguments.append("preserve_dst=True")
    return tvm.script.from_source(
        f"""
@T.prim_func
def atomic(destination: T.Buffer((128,), "float32"), value: T.Buffer((128,), "float32"),
           returned: T.Buffer((128,), "float32")):
    T.device_entry()
    warp = T.warp_id([{2 if race else 1}])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((128,), "float32", scope="shared", align=16)
    old = T.alloc_local(({width},), "float32")
    if warp == 0:
        for i in T.serial({width}):
            shared[lane * {width} + i] = destination[lane * {width} + i]
            old[i] = T.float32(-1)
        T.ptx["{spelling}"]({", ".join(arguments)})
        for i in T.serial({width}):
            returned[lane * {width} + i] = old[i]
            {"destination[lane * " + str(width) + " + i] = shared[lane * " + str(width) + " + i]" if pointer == "shared" else "T.evaluate(0)"}
    else:
        destination[lane * {width}] = T.float32(3)
""",
        {"T": T},
    )


def atomic_inputs():
    # Tiny inputs; normal cancellation to a subnormal; tie-to-even; signed
    # zero; infinities and NaNs. Old-value payloads must remain bit-exact.
    left = [1, 0x80000001, 0x00800001, 0x3F800000, 0x80000000, 0x7F800000, 0x7FC12345, 0x00800000]
    right = [1, 0x80000001, 0x80800000, 0x33800000, 0x80000000, 0xFF800000, 0x3F800000, 0x807FFFFF]
    return {
        # An odd period ensures predicated v2/v4 lanes exercise every pair.
        "destination": np.resize(np.array([*left, 0x3F800001], np.uint32), 128).view(np.float32),
        "value": np.resize(np.array([*right, 0x33800000], np.uint32), 128).view(np.float32),
        "returned": np.full(128, -1, np.float32),
    }


def assert_float_bits(actual, expected):
    nan = np.isnan(expected)
    np.testing.assert_array_equal(np.isnan(actual), nan)
    np.testing.assert_array_equal(actual[~nan].view(np.uint32), expected[~nan].view(np.uint32))


def addition_expected(*, flush=False):
    # Hand-derived RNE results, independent of both the engine and the host's
    # floating-point flush mode. NaN payloads are intentionally unspecified.
    bits = np.array(
        [2, 0x80000002, 1, 0x3F800000, 0x80000000, 0x7FC00000, 0x7FC00000, 1, 0x3F800002],
        np.uint32,
    )
    if flush:
        bits[[0, 1, 2, 7]] = [0, 0x80000000, 0, 0x00800000]
    return np.resize(bits, 128).view(np.float32)


@pytest.mark.parametrize("kind,width,space", ATOMIC_CASES)
def test_atomic_f32_noftz(kind, width, space, tmp_path):
    for noftz in (False, True):
        kernel = atomic_kernel(kind, width, space, noftz=noftz)
        inputs = atomic_inputs()
        for checker in (synccheck, racecheck):
            checker(kernel, {name: value.copy() for name, value in inputs.items()}).require_clean()
        module = numsim.transpile(kernel, cache_dir=tmp_path)
        result = numsim.Engine().run(module, {name: value.copy() for name, value in inputs.items()})
        selected = np.arange(128) < 32 * width
        if width > 1:
            selected &= np.arange(128) // width % 2 == 0
        flush = not noftz and not space.startswith("shared")
        left = inputs["destination"]
        expected = left.copy()
        expected[selected] = addition_expected(flush=flush)[selected]
        assert_float_bits(result.outputs["destination"], expected)
        returned = inputs["returned"].copy()
        if kind == "atom":
            returned[selected] = left[selected]
        np.testing.assert_array_equal(
            result.outputs["returned"].view(np.uint32), returned.view(np.uint32)
        )
        assert ("variant::Add<true>" in module.rust_source) is noftz


def bulk_kernel(*, early_read=False):
    return tvm.script.from_source(
        f"""
@T.prim_func
def bulk(source: T.Buffer((16,), "float32"), destination: T.Buffer((16,), "float32"),
         observed: T.Buffer((1,), "float32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((16,), "float32", scope="shared", align=16)
    if lane == 0:
        for i in T.serial(16):
            shared[i] = source[i]
        T.ptx.fence.proxy.async_.shared__cta()
        T.ptx["cp.reduce.async.bulk.global.shared::cta.bulk_group.add.noftz.f32"](
            destination.ptr_to([0]), shared.ptr_to([0]), T.uint32(64))
        T.ptx.cp.async_.bulk.commit_group()
        T.ptx.cp.async_.bulk.wait_group.read(0)
        {"observed[0] = destination[0]" if early_read else "T.evaluate(0)"}
        for i in T.serial(16):
            shared[i] = T.float32(99)
        T.ptx.cp.async_.bulk.wait_group(0)
        observed[0] = destination[0]
""",
        {"T": T},
    )


def test_bulk_f32_noftz(tmp_path):
    data = atomic_inputs()
    inputs = {
        "source": data["value"][:16],
        "destination": data["destination"][:16],
        "observed": np.zeros(1, np.float32),
    }
    kernel = bulk_kernel()
    for checker in (synccheck, racecheck):
        checker(kernel, {name: value.copy() for name, value in inputs.items()}).require_clean()
    result = numsim.Engine().run(
        numsim.transpile(kernel, cache_dir=tmp_path),
        {name: value.copy() for name, value in inputs.items()},
    )
    expected = addition_expected()[:16]
    assert_float_bits(result.outputs["destination"], expected)
    assert_float_bits(result.outputs["observed"], expected[:1])
    report = racecheck(bulk_kernel(early_read=True), inputs)
    assert report.verdict == "error", report.format()
    assert any(finding.details["access_pair"] == "write_read" for finding in report.findings), report.format()


@pytest.mark.parametrize("mutation", ["alignment", "race"])
def test_noftz_keeps_atomic_memory_checks(mutation):
    kernel = atomic_kernel(
        "sink", 4, "global", offset=int(mutation == "alignment"), race=mutation == "race"
    )
    report = racecheck(kernel, atomic_inputs())
    assert report.verdict == "error", report.format()
    assert (
        ("align" in report.format())
        if mutation == "alignment"
        else any(
            finding.details["access_pair"] in {"write_write", "read_write", "write_read"}
            for finding in report.findings
        )
    ), report.format()
