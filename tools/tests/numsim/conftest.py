from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--run-numsim-gpu",
        action="store_true",
        dest="run_numsim_gpu",
        default=True,
        help="run live same-kernel NumSim-vs-GPU microtests (default: enabled)",
    )
    parser.addoption(
        "--no-run-numsim-gpu",
        action="store_false",
        dest="run_numsim_gpu",
        help="skip live same-kernel NumSim-vs-GPU microtests",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if config.getoption("run_numsim_gpu"):
        return
    skip = pytest.mark.skip(reason="NumSim/GPU microtests disabled by --no-run-numsim-gpu")
    for item in items:
        if "numsim_gpu" in item.keywords:
            item.add_marker(skip)


@pytest.fixture
def gpu_runner(pytestconfig):
    """Share the ordinary device admission check and existing GPU runner."""
    from tests.numsim.microtests.harness import require_numsim_gpu, run_gpu_primfunc

    require_numsim_gpu(pytestconfig)
    return run_gpu_primfunc


@pytest.fixture
def expect_harness_surface():
    """Run one public harness action and check its returned value."""

    def expect(
        action: Callable[[], Any],
        check: Callable[[Any], None],
    ) -> Any:
        value = action()
        check(value)
        return value

    return expect


@pytest.fixture
def expect_harness_error():
    """Run one public harness action and check its typed failure."""

    def expect(
        action: Callable[[], Any],
        *,
        error: type[BaseException],
        match: str,
        check: Callable[[BaseException], None] | None = None,
    ) -> BaseException:
        with pytest.raises(error, match=match) as caught:
            action()
        if check is not None:
            check(caught.value)
        return caught.value

    return expect
