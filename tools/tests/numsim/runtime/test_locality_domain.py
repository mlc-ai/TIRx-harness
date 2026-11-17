from __future__ import annotations

import numpy as np
import pytest

import tvm_tirx
from tirx_harness import numsim
from tvm_tirx.script import tirx as T


def _locality_kernel(grid: int, cluster: int):
    source = f"""
@T.function
def locality_domains(
    domain: T.Tensor(({grid}, 64), "int32"), supplemental: T.Tensor(({grid}, 64), "int32")
):
    T.device_entry(launch=T.cuda.LaunchConfig(grid={grid}, block=64, cluster={cluster}))
    cta = T.cuda.block_idx("x")
    thread = T.cuda.thread_idx("x")
    domain[cta, thread] = T.cuda.mov_sreg(32, "locality_domain_id")
    supplemental[cta, thread] = T.cuda.mov_sreg(32, "locality_domain_supplemental_sm")
"""
    return tvm_tirx.script.from_source(source, extra_vars={"T": T})


@pytest.mark.parametrize("cluster", (1, 2, 4))
def test_locality_domain_registers_follow_ptx_placement_rules(tmp_path, cluster):
    grid = 8
    result = numsim.Engine().run(
        numsim.transpile(_locality_kernel(grid, cluster), cache_dir=tmp_path),
        {
            "domain": np.full((grid, 64), -1, np.int32),
            "supplemental": np.full((grid, 64), -1, np.int32),
        },
    )
    assert result.verdict == "clean"
    domain = result.outputs["domain"]
    supplemental = result.outputs["supplemental"]

    # PTX 10.36: every thread of a cluster reads one domain ID in [0, N).
    clusters = domain.reshape(grid // cluster, cluster * 64)
    assert (clusters == clusters[:, :1]).all()
    assert set(np.unique(domain)) == {0, 1}
    # PTX 10.37: the flag is 0 or 1 and uniform within a CTA, but CTAs of
    # one cluster may differ.
    assert (supplemental == supplemental[:, :1]).all()
    assert set(np.unique(supplemental)) == {0, 1}
    pairs = {(int(d), int(s)) for d, s in zip(domain[:, 0], supplemental[:, 0], strict=True)}
    if cluster == 1:
        assert pairs == {(0, 0), (0, 1), (1, 0), (1, 1)}
    else:
        flags = supplemental[:, 0].reshape(grid // cluster, cluster)
        assert (flags.min(axis=1) != flags.max(axis=1)).all()
