"""Same-case NumSim/GPU differential for the SM107-only tcgen05 forms.

Every case here reuses a runtime test's kernel and inputs. Those runtime tests
pin NumSim to an independent oracle; this file pins the same kernels to SM107
hardware, so the oracle, NumSim and the GPU must all agree.
"""

import numpy as np
import pytest

from tests.numsim.microtests.harness import require_numsim_gpu, run_paired_primfunc
from tests.numsim.runtime.test_launch_resource_facts import exclusive_lifecycle
from tests.numsim.runtime.test_tcgen05_block_tmem import block_tmem_case, fp4_tmem_case
from tests.numsim.runtime.test_tcgen05_ld_spcompress import compression_kernel
from tests.numsim.runtime.test_tcgen05_sparse_b16 import SPARSE_B16_CASES, sparse_float_case
from tests.numsim.runtime.test_tcgen05_sparse_k128 import SPARSE_K128_CASES, sparse_k128_case
from tests.numsim.runtime.test_tcgen05_sparse_narrow import SPARSE_NARROW_CASES, sparse_narrow_case
from tests.numsim.runtime.test_tcgen05_tf32 import TF32_SPARSE_CASES, tf32_case
from tests.numsim.runtime.test_tcgen05_ti16 import sparse_b16_case, ti16_kernel
from tests.numsim.runtime.test_tcgen_ashift import ashift_case
from tests.numsim.runtime.test_tcgen_collectors import collector_case
from tests.numsim.runtime.test_tcgen_lut_b import lut_b_case
from tests.numsim.support.execution import specialize, with_cuda_arch


def _cases():
    """(id, builder) pairs; each builder returns a kernel and its arguments."""

    def first_two(factory, *args, **kwargs):
        return lambda: factory(*args, **kwargs)[:2]

    for options in (
        dict(tmem_a=False, ws=False, slot="b", cta_group=1),
        dict(tmem_a=True, ws=False, slot="a", cta_group=2),
        dict(tmem_a=False, ws=True, slot="b2", cta_group=1),
    ):
        yield (
            "collector-" + "-".join(f"{key}={value}" for key, value in options.items()),
            first_two(collector_case, ("fill", "!discard", "use", "lastuse", ""), **options),
        )
    # K=32 block-scaled TMEM-A forms predate SM107; retarget them to check
    # that SM107 keeps their semantics.
    for formats in ((1, 0, 1), (2, 1, 3), (1, 3, 4), (2, 4, 5), (1, 5, 0)):
        yield (
            f"block_tmem-{formats}",
            lambda formats=formats: _retargeted(*block_tmem_case(*formats)[:2]),
        )
    for cta_group, tmem_a in ((1, False), (2, True)):
        yield (
            f"mxf8f6f4-k64-explicit-{cta_group}-tmem_a={tmem_a}",
            first_two(block_tmem_case, cta_group, 3 if tmem_a else 0, 1, tmem_a=tmem_a, k=64),
        )
    # K=64 TMEM A packs every format contiguously (16/12/8 words per row).
    for cta_group, a_format, b_format in ((1, 0, 1), (2, 1, 0), (1, 5, 3), (2, 4, 5)):
        yield (
            f"mxf8f6f4-k64-tmem_a-{cta_group}-{a_format}-{b_format}",
            first_two(block_tmem_case, cta_group, a_format, b_format, tmem_a=True, k=64),
        )
    for tmem_a in (False, True):
        yield f"mxf4-implicit-tmem_a={tmem_a}", first_two(fp4_tmem_case, tmem_a)
        yield (
            f"mxf8f6f4-k64-implicit-tmem_a={tmem_a}",
            first_two(
                block_tmem_case, 2, 3 if tmem_a else 0, 1, tmem_a=tmem_a, k=64, implicit_scale=True
            ),
        )
    for group, tmem, segment, half in (
        (1, False, 0, False),
        (1, True, 1, True),
        (2, True, 0, False),
    ):
        yield (
            f"lut_b-{group}-{tmem}-{segment}-{half}",
            first_two(lut_b_case, group, tmem, segment, half),
        )
    for group, tmem, segment in ((1, True, 1), (2, False, 1)):
        yield (
            f"lut_b_block-{group}-{tmem}-{segment}",
            first_two(lut_b_case, group, tmem, segment, block=True),
        )
    for cta_group, kind, sparse, bf16 in (
        (2, "f16", False, False),
        (1, "tf32", False, False),
        (2, "ti16", False, False),
        (1, "ti16", True, False),
        (2, "tf32", True, False),
        (2, "f16", True, True),
        (1, "f8f6f4", True, False),
        (2, "f8f6f4", True, False),
    ):
        yield (
            f"ashift-{cta_group}-{kind}-sparse={sparse}-bf16={bf16}",
            first_two(ashift_case, cta_group, followup=True, kind=kind, sparse=sparse, bf16=bf16),
        )
    for case in TF32_SPARSE_CASES:
        yield f"tf32_sparse-{case}", first_two(tf32_case, *case, int(not case[2]))
    for case in SPARSE_NARROW_CASES:
        yield f"sparse_narrow-{case}", first_two(sparse_narrow_case, *case)
    for case in SPARSE_K128_CASES:
        yield f"sparse_k128-{case}", first_two(sparse_k128_case, *case)
    for case in SPARSE_B16_CASES:
        yield f"sparse_b16-{case}", first_two(sparse_float_case, *case, selector=int(not case[2]))
    for cta_group, m, tmem_a, ws in (
        (1, 128, True, False),
        (1, 64, False, True),
        (2, 256, True, False),
    ):
        collectors = ""
        if cta_group == 2:
            collectors = ".collector::a::discard.collector::b::discard"
        yield (
            f"sparse_ti16-{cta_group}-{m}-{tmem_a}-{ws}",
            first_two(
                sparse_b16_case,
                m,
                tmem_a,
                ws,
                mask_form=ws,
                cta_group=cta_group,
                collectors=collectors,
            ),
        )
    for cta_group, m, tmem_a in ((1, 128, False), (1, 128, True), (2, 256, True)):
        yield f"ti16-{cta_group}-{m}-{tmem_a}", lambda c=cta_group, m=m, t=tmem_a: _ti16(c, m, t)
    # Restricted commit is covered by test_tcgen_commit_predicates; its
    # runtime oracle uses kind::i8, which ptxas rejects for sm_107a.
    for num, reduce, maximum, absolute, nan in (
        (4, False, True, False, False),
        (8, True, False, True, True),
        (128, True, True, False, False),
    ):
        yield (
            f"ld_spcompress-{num}-{reduce}-{maximum}-{absolute}-{nan}",
            lambda n=num, r=reduce, mx=maximum, a=absolute, na=nan: _spcompress(n, r, mx, a, na),
        )
    yield "exclusive_576", lambda: _exclusive_576("sm_107a")


def _ti16(cta_group, m, tmem_a):
    n = 16 * cta_group
    a = np.full((4, m, 16), 2047, np.int64)
    a[:, ::2, :] *= -1
    a[:, :, 0] //= 2
    a[1] //= 3
    b = np.full((n, 16), 2047, np.int64)
    b[:, 0] -= np.arange(n)
    seed = (np.arange(m * n, dtype=np.int64).reshape(m, n) + 2**31 - 1024).astype(np.int32)

    def encode(values):
        return (np.abs(values) | ((values < 0).astype(np.int64) << 15)).astype(np.uint16)

    collectors = ""
    if cta_group == 2:
        collectors = (".collector::a::discard" if m == 256 else "") + ".collector::b::discard"
    kernel = ti16_kernel(True, tmem_a, cta_group, m, collectors=collectors)
    return kernel, {
        "a": encode(a),
        "b": encode(b),
        "zero_mask": np.zeros(1, np.uint64),
        "metadata": np.zeros((2, 128, 2), np.uint32),
        "seed": seed,
        "out": np.zeros_like(seed),
    }


def _spcompress(num, reduce, maximum, absolute, nan):
    source = np.tile(np.array([-1, 4, -2, 3], np.float32), (32, num // 4))
    source[1, :4] = [np.nan, 4, np.nan, 3]
    count = (num + 31) // 32 + num // 2 + int(reduce)
    kernel = compression_kernel(num, reduce=reduce, maximum=maximum, absolute=absolute, nan=nan)
    return kernel, {"source": source, "output": np.zeros((32, count), np.uint32)}


def _exclusive_576(arch):
    kernel = with_cuda_arch(
        specialize(exclusive_lifecycle, {exclusive_lifecycle.params[0]: 576}), arch
    )
    return kernel, {"output": np.full(2, 99, dtype=np.uint32)}


def _retargeted(kernel, arguments):
    return with_cuda_arch(kernel, "sm_107a"), arguments


CASES = dict(_cases())


def _written_outputs(kernel, arguments):
    """Array parameters a kernel may write: everything but obvious inputs."""

    inputs = {"a", "b", "a_physical", "b_physical", "source", "seed", "metadata", "zero_mask"}
    return tuple(
        name
        for name, value in arguments.items()
        if isinstance(value, np.ndarray) and name not in inputs and not name.startswith("scale")
    )


@pytest.mark.numsim_gpu
@pytest.mark.parametrize("case", sorted(CASES))
def test_sm107_tcgen05_form_matches_gpu(case, pytestconfig, tmp_path):
    require_numsim_gpu(pytestconfig, min_nvrtc=(13, 4))
    import torch

    if torch.cuda.get_device_capability() != (10, 7):
        pytest.skip("SM107 tcgen05 forms require an SM107 GPU")
    kernel, arguments = CASES[case]()
    run_paired_primfunc(
        kernel,
        arguments,
        outputs=_written_outputs(kernel, arguments),
        cache_dir=tmp_path,
        arch="sm_107a",
    )
