from __future__ import annotations

import pytest

from tests.numsim.support.kernels import dsmem_copy_remote_cta, tma_copy_cluster_multicast
from tirx_harness.numsim.checker_runner import transpile_native_checker_artifact


@pytest.mark.parametrize("checker", ["synccheck", "racecheck"])
@pytest.mark.parametrize(
    "kernel",
    [tma_copy_cluster_multicast, dsmem_copy_remote_cta],
    ids=["literal-tma-cta-mask", "literal-dsmem-remote-cta"],
)
def test_checker_transpilation_accepts_literal_tile_protocol_config(checker, kernel, tmp_path):
    module = transpile_native_checker_artifact(checker, kernel, cache_dir=tmp_path)

    assert len(module.spec.kernels) == 1
    assert module.rust_source
