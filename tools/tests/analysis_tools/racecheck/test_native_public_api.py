from __future__ import annotations

import inspect

import numpy as np
import pytest
from tvm.script import tirx as T

from tirx_harness import numsim
from tirx_harness.numsim.cases import _descriptor_storage
from tirx_harness import racecheck as facade_racecheck
from tirx_harness import synccheck as facade_synccheck
from tirx_harness.numsim.checker_runner import transpile_native_checker_artifact

_DESCRIPTOR_SOURCE = np.arange(4, dtype=np.float32)


@T.prim_func
def native_public_api_kernel():
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((32,), "int32", scope="shared")
    shared[lane] = lane


@T.prim_func
def native_public_griddep_wait_kernel(output: T.Buffer((1,), "int32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    T.ptx.griddepcontrol.wait()
    if lane == 0:
        output[0] = 1


@T.prim_func
def native_public_descriptor_storage_kernel(
    descriptor_storage: T.Buffer((128,), "uint8"), output: T.Buffer((1,), "int32")
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    if lane == 0:
        output[0] = T.cast(descriptor_storage[0], "int32")


@T.prim_func
def native_public_scalar_kernel(mode: T.int32, output: T.Buffer((1,), "int32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    if lane == 0:
        output[0] = mode


def _descriptor_storage_binding() -> np.ndarray:
    return _descriptor_storage(
        storage=np.zeros(128, dtype=np.uint8),
        slots={
            0: numsim.TensorMap(
                base=_DESCRIPTOR_SOURCE,
                global_shape=(4, 1),
                global_strides=(16,),
                box_shape=(4, 1),
                element_strides=(1, 1),
            ).numpy()
        },
    )


def test_public_checker_signatures_are_kernel_and_inputs() -> None:
    for checker in (facade_synccheck, facade_racecheck):
        assert tuple(inspect.signature(checker).parameters) == ("kernel", "inputs")


@pytest.mark.parametrize("checker", [facade_synccheck, facade_racecheck])
@pytest.mark.parametrize("option", ["max_while_iters", "max_workers", "cache_dir"])
def test_public_checkers_reject_legacy_and_internal_options(checker, option: str) -> None:
    with pytest.raises(TypeError, match=option):
        checker(native_public_api_kernel, {}, **{option: 1})


def test_production_racecheck_artifact_omits_unused_synccheck_entrypoint(tmp_path) -> None:
    module = transpile_native_checker_artifact(
        "racecheck",
        native_public_api_kernel,
        cache_dir=tmp_path,
    )

    assert "fn native_racecheck_phase(" in module.rust_source
    assert "fn native_synccheck_phase(" not in module.rust_source
    assert "\nfn run(" not in module.rust_source
    native = module.load()
    assert hasattr(native, "_native_racecheck_phase")
    assert not hasattr(native, "_native_synccheck_phase")
    assert not hasattr(native, "run")


def test_production_synccheck_artifact_omits_unused_racecheck_entrypoint(tmp_path) -> None:
    module = transpile_native_checker_artifact(
        "synccheck",
        native_public_api_kernel,
        cache_dir=tmp_path,
    )

    assert "fn native_synccheck_phase(" in module.rust_source
    assert "fn native_racecheck_phase(" not in module.rust_source
    assert "\nfn run(" not in module.rust_source
    native = module.load()
    assert hasattr(native, "_native_synccheck_phase")
    assert not hasattr(native, "_native_racecheck_phase")
    assert not hasattr(native, "run")


def test_public_interface_executes_native() -> None:
    report = facade_racecheck(native_public_api_kernel, {})
    payload = report.to_dict()
    assert report.verdict == "clean"
    assert payload["engine"] == "native"
    assert payload["native"]["execution_model"] == "direct_online_vc"
    assert payload["native"]["incomplete"] == []
    assert report.findings == []


def test_public_synccheck_interface_executes_native() -> None:
    report = facade_synccheck(native_public_api_kernel, {})
    payload = report.to_dict()
    assert report.verdict == "clean"
    assert payload["engine"] == "native"
    assert payload["native"]["incomplete"] == []
    assert report.findings == []


def test_public_synccheck_assumes_external_grid_dependency_is_satisfied() -> None:
    report = facade_synccheck(
        native_public_griddep_wait_kernel,
        {"output": np.zeros((1,), dtype=np.int32)},
    )

    assert report.verdict == "clean"
    assert report.findings == []


@pytest.mark.parametrize("checker", [facade_synccheck, facade_racecheck])
def test_public_checkers_accept_tensor_map_descriptor_storage(checker) -> None:
    report = checker(
        native_public_descriptor_storage_kernel,
        {
            "descriptor_storage": _descriptor_storage_binding(),
            "output": np.zeros((1,), dtype=np.int32),
        },
    )

    assert report.verdict == "clean"
    assert report.findings == []


@pytest.mark.parametrize("checker", [facade_synccheck, facade_racecheck])
def test_public_checkers_reject_descriptor_storage_for_scalar(checker) -> None:
    with pytest.raises(
        numsim.NumSimExecutionError,
        match="native analysis scalar input 'mode' has a buffer value",
    ):
        checker(
            native_public_scalar_kernel,
            {
                "mode": _descriptor_storage_binding(),
                "output": np.zeros((1,), dtype=np.int32),
            },
        )


def test_report_scope_line_matches_checked_spaces() -> None:
    report = facade_racecheck(native_public_api_kernel, {})
    text = report.format()
    # The scoped global model race-classifies global accesses, and the scope
    # line must not carry the pre-global-model disclaimer claiming otherwise.
    assert "race conflicts checked in global, shared, tmem" in text
    assert "not race-classified" not in text
