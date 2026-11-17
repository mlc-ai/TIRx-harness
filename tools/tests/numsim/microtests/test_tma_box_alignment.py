"""Hardware differential for 16-byte-aligned TMA bounding-box starts.

Unaligned tiled and im2col-channel starts trap with an illegal instruction
on SM107 (PTX 5.5.3.1 for tiled mode) and are
covered by NumSim-only rejection tests; this file runs only aligned starts.
"""

import numpy as np
import pytest

from tests.numsim.microtests.harness import (
    PairedTensorMap,
    require_numsim_gpu,
    run_paired_primfunc,
)
from tests.numsim.runtime.test_tma_box_alignment import (
    ALIGNED,
    box_kernel,
    box_map,
    expected_box,
    expected_im2col_load,
    im2col_box_kernel,
    im2col_box_map,
)


@pytest.mark.numsim_gpu
@pytest.mark.parametrize("route", ["cta", "cluster"])
def test_aligned_tile_load_starts_match_gpu(pytestconfig, tmp_path, route):
    require_numsim_gpu(pytestconfig)
    kernel = box_kernel(route)
    for x in ALIGNED:
        for y in (0, 1, -1):
            source, metadata = box_map(route)
            result = run_paired_primfunc(
                kernel,
                {
                    "descriptor": PairedTensorMap(source, **metadata),
                    "x": x,
                    "y": y,
                    "output": np.zeros(8, np.uint32),
                },
                outputs=("output",),
                cache_dir=tmp_path,
            )
            np.testing.assert_array_equal(
                result.gpu_outputs["output"],
                expected_box(source, x, (y, y + 1)),
                err_msg=f"{x, y}",
            )


@pytest.mark.numsim_gpu
def test_aligned_im2col_channel_starts_match_gpu(pytestconfig, tmp_path):
    require_numsim_gpu(pytestconfig)
    kernel = im2col_box_kernel("load")
    for c in (0, 4):
        source, metadata = im2col_box_map("load")
        result = run_paired_primfunc(
            kernel,
            {
                "tmap": PairedTensorMap(source, **metadata),
                "c": c,
                "out": np.zeros(48, np.uint32),
            },
            outputs=("out",),
            cache_dir=tmp_path,
        )
        np.testing.assert_array_equal(
            result.gpu_outputs["out"], expected_im2col_load(c), err_msg=f"c={c}"
        )
