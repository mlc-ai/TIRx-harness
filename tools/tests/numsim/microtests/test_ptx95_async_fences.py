"""Hardware pairing for the PTX ISA 9.5 restricted bulk wait and alias read fence.

Only race-free variants run here: a restricted wait followed directly by a
generic access to the copy's bytes is a race whose hardware result is not an
oracle for anything.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.numsim.microtests.harness import require_numsim_gpu, run_gpu_primfunc
from tests.numsim.runtime.test_bulk_wait_group_proxy_async import (
    PLAIN,
    PLAIN_READ,
    RESTRICTED,
    RESTRICTED_THEN_PLAIN,
    RESTRICTED_THEN_PROXY_FENCE,
    _global_read_inputs,
    _source,
    restricted_wait_keeps_newest_groups_pending,
    s2g_then_async_g2s_round_trip,
    s2g_then_generic_global_read,
    s2g_then_generic_shared_write,
)
from tests.numsim.runtime.test_fence_alias_read_release import (
    _inputs as _alias_fence_inputs,
)
from tests.numsim.runtime.test_fence_alias_read_release import read_release_handoff
from tirx_harness import numsim

# PTX ISA 9.5 spellings need an NVRTC that accepts PTX 9.5.
_MIN_NVRTC = (13, 5)


def _cases():
    shared_write_inputs = lambda mode: {  # noqa: E731
        "mode": np.int32(mode),
        "source": _source(),
        "destination": np.zeros(16, np.uint8),
    }
    pending_inputs = {
        "group_read": np.int32(0),
        "source": _source(32),
        "destination": np.zeros(32, np.uint8),
        "output": np.zeros(16, np.uint8),
    }
    return [
        *(
            pytest.param(
                s2g_then_generic_global_read,
                _global_read_inputs(mode),
                ("destination", "output"),
                id=f"global-read-{name}",
            )
            for mode, name in (
                (PLAIN, "plain"),
                (RESTRICTED_THEN_PROXY_FENCE, "restricted-proxy-fence"),
                (RESTRICTED_THEN_PLAIN, "restricted-then-plain"),
            )
        ),
        *(
            pytest.param(
                s2g_then_generic_shared_write,
                shared_write_inputs(mode),
                ("destination",),
                id=f"shared-write-{name}",
            )
            for mode, name in (
                (PLAIN, "plain"),
                (PLAIN_READ, "plain-read"),
                (RESTRICTED_THEN_PROXY_FENCE, "restricted-proxy-fence"),
            )
        ),
        *(
            pytest.param(
                s2g_then_async_g2s_round_trip,
                _global_read_inputs(mode),
                ("destination", "output"),
                id=f"async-round-trip-{name}",
            )
            for mode, name in ((PLAIN, "plain"), (RESTRICTED, "restricted"))
        ),
        pytest.param(
            restricted_wait_keeps_newest_groups_pending,
            pending_inputs,
            ("destination", "output"),
            id="restricted-pending-one",
        ),
        pytest.param(
            read_release_handoff,
            _alias_fence_inputs(1),
            ("data", "flag", "output"),
            id="alias-read-release-handoff",
        ),
    ]


def _numsim_arguments(inputs):
    # Scalar slots bind NumPy scalars; `np.copy` would turn them into 0-d arrays.
    return {name: value if np.ndim(value) == 0 else value.copy() for name, value in inputs.items()}


def _gpu_arguments(inputs):
    return {
        name: int(value) if np.ndim(value) == 0 else value.copy() for name, value in inputs.items()
    }


@pytest.mark.numsim_gpu
@pytest.mark.parametrize(("kernel", "inputs", "outputs"), _cases())
def test_ptx95_async_fences_match_gpu(pytestconfig, tmp_path, kernel, inputs, outputs):
    require_numsim_gpu(pytestconfig, min_nvrtc=_MIN_NVRTC)
    import torch

    major, minor = torch.cuda.get_device_capability()
    expected = numsim.Engine().run(
        numsim.transpile(kernel, cache_dir=tmp_path),
        _numsim_arguments(inputs),
    )
    device = run_gpu_primfunc(
        kernel, _gpu_arguments(inputs), outputs=outputs, arch=f"sm_{major}{minor}a"
    )
    for name in outputs:
        np.testing.assert_array_equal(device[name], expected.outputs[name], err_msg=name)
