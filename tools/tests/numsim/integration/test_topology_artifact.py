from __future__ import annotations

import numpy as np
import pytest

from tirx_harness import numsim
from tirx_harness.numsim.transpiler.frontend import analyze
from tvm.script import tirx as T


@T.prim_func
def too_many_warps(output: T.Buffer((1,), "int32")):
    T.device_entry()
    warp = T.warp_id([33])
    lane = T.lane_id([32])
    if (warp == 0) and (lane == 0):
        output[0] = 1


@T.prim_func
def too_many_cluster_ctas(output: T.Buffer((1,), "int32")):
    T.device_entry()
    _cluster = T.cluster_id([1])
    cta = T.cta_id_in_cluster([65])
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    if (cta == 0) and (lane == 0):
        output[0] = 1


@T.prim_func
def cluster_cta_and_pair_coordinates(output: T.Buffer((8, 2), "int32")):
    T.device_entry()
    cta = T.cta_id_in_cluster([8])
    pair = T.cta_id_in_pair()
    lane = T.lane_id([32])
    if lane == 0:
        output[cta, 0] = cta
        output[cta, 1] = pair


def test_artifact_launch_rejects_topologies_outside_engine_representation(tmp_path):
    output = np.zeros(1, dtype=np.int32)

    too_many_warps_module = numsim.transpile(too_many_warps, cache_dir=tmp_path)
    with pytest.raises(numsim.NumSimBuildError, match="warps_per_cta=33.*maximum 32"):
        numsim.Engine().run(too_many_warps_module, {"output": output})

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
    assert "ctx.cta_id_in_cluster() as i64" in module.rust_source
    assert "(ctx.cta_id_in_cluster() % 2) as i64" in module.rust_source
