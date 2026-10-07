"""Keep resolved instruction reports as passing regression gates."""

import ml_dtypes
import numpy as np
import pytest
from tvm.backend.cuda.cpp.descriptors import encode_instr_descriptor_dense_uint32

import tirx_kernels.tirx_lite as txl
from tirx_harness import synccheck


@txl.kernel(warps=1, arch="sm_100a", grid=1)
def bulk_bf16_reduction(out: txl.gptr[txl.bf16]):
    smem = txl.smem_pool()
    source = smem.alloc((32,), txl.bf16, align=128)
    lane = txl.lane_id()
    txl.ptx.st.shared.b16(source.ptr_to([lane]), txl.uint16(0x3F80))
    txl.ptx.fence.proxy.async_.shared__cta()
    txl.ptx.bar.sync(txl.uint32(0))
    with txl.If(lane == 0), txl.Then():
        txl.ptx["cp.reduce.async.bulk.global.shared::cta.bulk_group.add.noftz.bf16"](
            out.ptr_to([0]), source.ptr_to([0]), txl.uint32(64)
        )
        txl.ptx.cp.async_.bulk.commit_group()
        txl.ptx.cp.async_.bulk.wait_group.read(0)


@txl.kernel(warps=1, arch="sm_100a", grid=1)
def packed_bf16_vector_reduction(out: txl.gptr[txl.bf16]):
    with txl.If(txl.thread_id() == 0), txl.Then():
        txl.ptx["red.global.v2.bf16x2.add.noftz"](
            out.ptr_to([0]), txl.uint32(0x3F803F80), txl.uint32(0x3F803F80)
        )


@txl.kernel(warps=1, arch="sm_100a", grid=1)
def fractional_cache_policy():
    policy = txl.local_scalar(txl.u64)
    txl.ptx.createpolicy.fractional.L2__evict_first.b64(policy)


@pytest.mark.parametrize(
    "kernel, output_elements",
    [
        (bulk_bf16_reduction, 32),
        (packed_bf16_vector_reduction, 4),
        (fractional_cache_policy, 0),
    ],
    ids=["cp_reduce_bulk_bf16", "red_vec_packed_bf16", "createpolicy_fractional"],
)
def test_reported_instruction_support(kernel, output_elements):
    bindings = (
        {"out": np.zeros(output_elements, dtype=ml_dtypes.bfloat16)} if output_elements else {}
    )
    report = synccheck(kernel.func, bindings)
    assert report.verdict == "clean", report.to_dict()


def make_ws_probe(predicated):
    instruction_descriptor = encode_instr_descriptor_dense_uint32(
        M=64,
        N=16,
        K=16,
        d_dtype="float32",
        a_dtype="float16",
        b_dtype="float16",
        trans_a=False,
        trans_b=False,
        cta_group=1,
    )

    @txl.kernel(warps=1, arch="sm_100a", grid=1)
    def ws_probe():
        smem = txl.smem_pool()
        a = smem.alloc((64, 64), txl.f16, align=1024, swizzle=txl.SW128B)
        b = smem.alloc((16, 64), txl.f16, align=1024, swizzle=txl.SW128B)
        a_desc, a_off = a.encode(major="k", mma_k=16)
        b_desc, b_off = b.encode(major="k", mma_k=16)
        lane = txl.lane_id()
        tmem_slot = smem.alloc((1,), txl.u32, align=16)
        txl.ptx["tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32"](
            txl.address_of(tmem_slot[0]), txl.uint32(32)
        )
        txl.ptx["tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned"]()
        txl.ptx.bar.sync(txl.uint32(0))
        tmem_base = txl.local_scalar(txl.u32)
        txl.ptx.ld.shared.b32(tmem_base, txl.address_of(tmem_slot[0]))
        args = (
            tmem_base,
            a_desc + a_off(0),
            b_desc + b_off(0),
            txl.uint32(instruction_descriptor),
            txl.cast(0 != 0, "bool"),
            txl.uint64(0),
        )
        if predicated:
            txl.ptx["tcgen05.mma.ws.cta_group::1.kind::f16"](
                *args, pred=txl.cast(lane == 0, "bool")
            )
        else:
            with txl.If(lane == 0), txl.Then():
                txl.ptx["tcgen05.mma.ws.cta_group::1.kind::f16"](*args)
        txl.ptx.bar.sync(txl.uint32(0))
        txl.ptx["tcgen05.dealloc.cta_group::1.sync.aligned.b32"](tmem_base, txl.uint32(32))

    return ws_probe


def test_ws_instruction_predication_matches_branch_election():
    reports = [synccheck(make_ws_probe(predicated).func) for predicated in (False, True)]
    # Shared operands are deliberately uninitialized. Matching the individual
    # reads also checks that predication elects one issuer.
    for report in reports:
        assert report.verdict == "review", report.to_dict()
        assert all(f.kind == "uninitialized_read" for f in report.findings), report.to_dict()
    assert sorted((f.kind, f.message) for f in reports[0].findings) == sorted(
        (f.kind, f.message) for f in reports[1].findings
    )
