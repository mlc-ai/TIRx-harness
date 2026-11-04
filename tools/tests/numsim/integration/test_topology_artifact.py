from __future__ import annotations

import numpy as np
import pytest

from tirx_harness import numsim
from tirx_harness.numsim.transpiler.frontend import analyze
from tvm_tirx.script import tirx as T


@T.function
def too_many_cluster_ctas(output: T.Tensor((1,), "int32")):
    T.device_entry(launch=T.cuda.LaunchConfig(grid=65, block=32, cluster=65))
    cta = T.cuda.cluster_cta_id("x")
    lane = T.cuda.lane_id()
    if (cta == 0) and (lane == 0):
        output[0] = 1


@T.function
def cluster_cta_and_pair_coordinates(output: T.Tensor((8, 2), "int32")):
    T.device_entry(launch=T.cuda.LaunchConfig(grid=8, block=32, cluster=8))
    cta = T.cuda.cluster_cta_id("x")
    pair = T.cuda.cta_pair_id()
    lane = T.cuda.lane_id()
    if lane == 0:
        output[cta, 0] = cta
        output[cta, 1] = pair


def test_artifact_launch_rejects_topologies_outside_engine_representation(tmp_path):
    output = np.zeros(1, dtype=np.int32)

    too_many_ctas_module = numsim.transpile(too_many_cluster_ctas, cache_dir=tmp_path)
    with pytest.raises(numsim.NumSimBuildError, match="ctas_per_cluster=65.*maximum 64"):
        numsim.Engine().run(too_many_ctas_module, {"output": output})


def test_cta_pair_uses_two_cta_residue_without_reducing_cluster_topology(tmp_path):
    spec = analyze(cluster_cta_and_pair_coordinates)
    assert spec.topology.clusters == 1
    assert spec.topology.ctas_per_cluster == 8

    output = np.zeros((8, 2), dtype=np.int32)
    expected = np.stack([np.arange(8, dtype=np.int32), np.arange(8, dtype=np.int32) % 2], axis=1)

    module = numsim.transpile(cluster_cta_and_pair_coordinates, cache_dir=tmp_path)
    result = numsim.Engine().run(module, {"output": output})

    np.testing.assert_array_equal(result.outputs["output"], expected)
    assert "(ctx.cta_id_in_cluster()) as i32" in module.rust_source
    assert "(ctx.cta_id_in_cluster() % 2) as i32" in module.rust_source
