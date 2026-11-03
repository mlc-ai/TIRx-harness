"""PTX tcgen05 thread-fence ordering, exercised through public Racecheck."""

from __future__ import annotations

import pytest

from tvm_tirx.tirx.layout import Axis, ComposeLayout, R, S, TileLayout

_MMA_F16_32B = ComposeLayout(3, 1, 3, TileLayout(S[(128,)]))
_REPLICATED_CP_LAYOUT = TileLayout(
    S[(32, 4) : (1 @ Axis.get("TLane"), 1 @ Axis.get("TCol"))] + R[4 : 32 @ Axis.get("TLane")]
)


@pytest.fixture(scope="module")
def native_cache_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("native-racecheck-tcgen-thread-fence")
