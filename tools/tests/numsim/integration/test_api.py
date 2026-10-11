from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from threading import Event

import numpy as np
import pytest
from threadpoolctl import threadpool_info

from tirx_harness import numsim
from tirx_harness.numsim import api as numsim_api
from tirx_harness.numsim.api import ExecutionSubset
from tirx_harness.numsim.bindings import prepare_bindings


def test_execution_subset_uses_subset_api_name():
    subset = ExecutionSubset(cluster_ids=[0, 2])
    assert subset.to_payload() == {"cluster_ids": [0, 2], "cta_ids": None}


def test_execution_subset_is_not_publicly_exported():
    assert not hasattr(numsim, "ExecutionSubset")
    assert not hasattr(numsim, "ExecutionSubsetSelection")
    assert "ExecutionSubset" not in numsim.__all__
    assert "ExecutionSubsetSelection" not in numsim.__all__


def test_multi_kernel_subsets_are_phase_indexed_and_never_broadcast():
    payload = numsim_api._execution_subset_payload(
        {0: ExecutionSubset(cluster_ids=[2, 0]), 2: ExecutionSubset(cta_ids=[7])},
        kernel_count=3,
    )

    assert payload == [
        {"cluster_ids": [0, 2], "cta_ids": None},
        None,
        {"cluster_ids": None, "cta_ids": [7]},
    ]
    with pytest.raises(ValueError, match="not broadcast"):
        numsim_api._execution_subset_payload(ExecutionSubset(cluster_ids=[0]), kernel_count=2)


def test_phase_subset_indices_and_ids_fail_closed():
    with pytest.raises(ValueError, match="outside"):
        numsim_api._execution_subset_payload({2: ExecutionSubset(cluster_ids=[0])}, kernel_count=2)
    with pytest.raises(TypeError, match="phase indices"):
        numsim_api._execution_subset_payload(
            {True: ExecutionSubset(cluster_ids=[0])}, kernel_count=2
        )
    with pytest.raises(ValueError, match="duplicate"):
        numsim_api._execution_subset_payload(
            {0: ExecutionSubset(cluster_ids=[1, 1])}, kernel_count=2
        )


def test_engine_worker_configuration_is_explicit_and_validated(monkeypatch):
    assert numsim.Engine().max_workers == 8
    assert numsim.Engine(max_workers=4).max_workers == 4

    monkeypatch.setattr("os.cpu_count", lambda: 6)
    assert numsim.Engine(max_workers="auto").max_workers == 6

    with pytest.raises(ValueError, match="positive"):
        numsim.Engine(max_workers=0)
    with pytest.raises(TypeError, match="positive integer"):
        numsim.Engine(max_workers=True)


def _blas_pool_threads():
    return {
        pool["filepath"]: pool["num_threads"]
        for pool in threadpool_info()
        if pool["user_api"] == "blas"
    }


def test_native_execution_restores_blas_threads_after_failure():
    before = _blas_pool_threads()
    assert before, "NumPy's BLAS backend must be loaded"
    with pytest.raises(RuntimeError, match="native failure"):
        with numsim_api._blas_thread_context():
            assert set(_blas_pool_threads().values()) == {1}
            raise RuntimeError("native failure")
    assert _blas_pool_threads() == before


def test_overlapping_native_executions_share_blas_limit_until_last_exit():
    before = _blas_pool_threads()
    first_entered, second_entered, first_exited = Event(), Event(), Event()

    def first():
        with numsim_api._blas_thread_context():
            first_entered.set()
            assert second_entered.wait(10)
        first_exited.set()

    def second():
        assert first_entered.wait(10)
        with numsim_api._blas_thread_context():
            second_entered.set()
            assert first_exited.wait(10)
            assert set(_blas_pool_threads().values()) == {1}

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(first), pool.submit(second)]
        for future in futures:
            future.result(timeout=20)
    assert _blas_pool_threads() == before


def test_engine_native_loop_policy_is_explicit_and_validated():
    defaults = numsim.Engine()
    assert defaults.native_loop_iteration_budget == 1_000_000
    assert defaults.native_loop_reschedule_quantum == 64

    configured = numsim.Engine(
        native_loop_iteration_budget=2_000_000, native_loop_reschedule_quantum=17
    )
    assert configured.native_loop_iteration_budget == 2_000_000
    assert configured.native_loop_reschedule_quantum == 17

    with pytest.raises(ValueError, match="native_loop_iteration_budget must be positive"):
        numsim.Engine(native_loop_iteration_budget=0)
    with pytest.raises(TypeError, match="native_loop_iteration_budget must be a positive integer"):
        numsim.Engine(native_loop_iteration_budget=True)
    with pytest.raises(ValueError, match="native_loop_reschedule_quantum must be positive"):
        numsim.Engine(native_loop_reschedule_quantum=0)
    with pytest.raises(
        TypeError, match="native_loop_reschedule_quantum must be a positive integer"
    ):
        numsim.Engine(native_loop_reschedule_quantum=1.5)


def test_result_assert_close_reports_first_mismatch():
    result = numsim.NumSimResult({"out": np.array([1.0, 3.0], dtype=np.float32)})
    with pytest.raises(AssertionError, match="first mismatch"):
        result.assert_close({"out": np.array([1.0, 2.0], dtype=np.float32)})


@pytest.mark.parametrize("dtype", [np.int32, np.uint64])
def test_integer_comparison_is_exact_by_default(dtype):
    result = numsim.NumSimResult({"out": np.array([100_000], dtype=dtype)})

    report = numsim.compare(result, {"out": np.array([100_001], dtype=dtype)})

    assert not report.ok
    assert report.mismatches[0].index == (0,)


def test_comparison_decodes_declared_bfloat16_backing():
    values = np.array([1.0, -2.5], dtype=np.float32)
    bits = values.view(np.uint32)
    encoded = (bits >> np.uint32(16)).astype(np.uint16)

    report = numsim.compare(
        numsim.NumSimResult({"out": encoded}),
        {"out": values},
        tolerances={"out": numsim.ComparisonSpec(rtol=0.0, atol=0.0, actual_encoding="bfloat16")},
    )

    assert report.ok


def test_comparison_decodes_declared_bfloat16_byte_carrier():
    values = np.array([1.0, -2.5], dtype=np.float32)
    bits = values.view(np.uint32)
    encoded = (bits >> np.uint32(16)).astype(np.uint16).view(np.uint8)

    report = numsim.compare(
        numsim.NumSimResult({"out": encoded}),
        {"out": values},
        tolerances={
            "out": numsim.ComparisonSpec(
                rtol=0.0,
                atol=0.0,
                actual_encoding="bfloat16",
            )
        },
    )

    assert report.ok


def test_comparison_rejects_odd_bfloat16_byte_carrier():
    with pytest.raises(numsim.NumSimExecutionError, match="even byte count"):
        numsim.compare(
            numsim.NumSimResult({"out": np.zeros(3, dtype=np.uint8)}),
            {"out": np.zeros(1, dtype=np.float32)},
            tolerances={"out": numsim.ComparisonSpec(actual_encoding="bfloat16")},
        )


def test_comparison_regions_map_physical_storage_to_logical_reference():
    actual = np.full((2, 6), -99, dtype=np.int32)
    expected = np.full((2, 6), 77, dtype=np.int32)
    actual[0, :2] = [3, 4]
    expected[0, 3:5] = [3, 4]
    spec = numsim.ComparisonSpec(
        rtol=0.0,
        atol=0.0,
        regions=(numsim.ComparisonRegion(actual=(0, slice(0, 2)), expected=(0, slice(3, 5))),),
    )

    assert numsim.compare(
        numsim.NumSimResult({"out": actual}), {"out": expected}, tolerances={"out": spec}
    ).ok

    actual[0, 1] = 5
    report = numsim.compare(
        numsim.NumSimResult({"out": actual}), {"out": expected}, tolerances={"out": spec}
    )
    assert not report.ok
    assert report.mismatches[0].index == (0, 1)


def test_compare_rejects_empty_expected_outputs():
    with pytest.raises(numsim.NumSimExecutionError, match="must not be empty"):
        numsim.compare(numsim.NumSimResult({}), {})


def test_compare_rejects_unknown_comparison_specs():
    with pytest.raises(numsim.NumSimExecutionError, match="unknown expected outputs"):
        numsim.compare(
            numsim.NumSimResult({"output": np.zeros(1, dtype=np.float32)}),
            {"output": np.zeros(1, dtype=np.float32)},
            tolerances={"typo": numsim.ComparisonSpec()},
        )


def test_compare_rejects_zero_element_default_region():
    with pytest.raises(numsim.NumSimExecutionError, match="selects zero elements"):
        numsim.compare(
            numsim.NumSimResult({"output": np.zeros(0, dtype=np.float32)}),
            {"output": np.zeros(0, dtype=np.float32)},
        )


def test_compare_rejects_zero_element_explicit_region():
    with pytest.raises(numsim.NumSimExecutionError, match="selects zero elements"):
        numsim.compare(
            numsim.NumSimResult({"output": np.zeros(4, dtype=np.float32)}),
            {"output": np.zeros(4, dtype=np.float32)},
            tolerances={
                "output": numsim.ComparisonSpec(
                    regions=(numsim.ComparisonRegion(actual=(slice(0, 0),)),)
                )
            },
        )


@pytest.mark.parametrize(
    ("kwargs", "error", "message"),
    [
        ({"rtol": -1.0}, ValueError, "finite non-negative"),
        ({"atol": float("inf")}, ValueError, "finite non-negative"),
        ({"equal_nan": "false"}, TypeError, "equal_nan must be a bool"),
        ({"regions": []}, TypeError, "regions must be a tuple"),
    ],
)
def test_comparison_spec_rejects_permissive_field_values(kwargs, error, message):
    with pytest.raises(error, match=message):
        numsim.ComparisonSpec(**kwargs)


def test_comparison_region_normalizes_integer_like_indices_and_rejects_fractional_indices():
    region = numsim.ComparisonRegion(actual=(np.int32(1), slice(np.int64(2), None, 1)))

    assert region.actual == (1, slice(2, None, 1))
    with pytest.raises(TypeError, match="index must be an integer"):
        numsim.ComparisonRegion(actual=(1.5,))


class _FrozenNoOpEngine:
    def __init__(self, source: np.ndarray) -> None:
        self.source = source
        self.executed = False
        self.observed_source: np.ndarray | None = None

    def _prepare_execution(self, module, inputs, *, outputs, assumptions=None):
        del module, assumptions
        assert outputs == ("output",)
        return numsim_api._PreparedExecution(
            bindings=prepare_bindings(inputs),
            output_names=frozenset({"output"}),
            external_names={"output": "output"},
            assumptions={"external_grid_dependencies_satisfied": []},
        )

    def _execute_prepared(self, module, execution, *, subset):
        del module, subset
        self.executed = True
        self.observed_source = self.source.copy()
        allocation_bytes = [allocation.data for allocation in execution.bindings.allocations]
        outputs = execution.bindings.apply_allocation_bytes(
            allocation_bytes, output_names={"output"}
        )
        return numsim.NumSimResult(outputs)


def test_run_case_requires_reference_keys_to_name_selected_outputs(monkeypatch):
    source = np.array([3], dtype=np.int32)
    engine = _FrozenNoOpEngine(source)
    case = numsim.NumSimCase(
        kernel=object(),
        args={
            "source": source,
            "output": np.zeros(1, dtype=np.int32),
        },
        outputs=("output",),
        reference=lambda: {"wrong_name": np.zeros(1, dtype=np.int32)},
    )
    monkeypatch.setattr(numsim_api, "transpile", lambda kernel, *, precision: kernel)

    with pytest.raises(numsim.NumSimExecutionError, match="must name selected"):
        numsim.run_case(case, engine=engine)

    assert not engine.executed


def test_run_case_rejects_empty_reference_before_execution(monkeypatch):
    source = np.array([3], dtype=np.int32)
    engine = _FrozenNoOpEngine(source)
    case = numsim.NumSimCase(
        kernel=object(),
        args={
            "source": source,
            "output": np.zeros(1, dtype=np.int32),
        },
        outputs=("output",),
        reference=lambda: {},
    )
    monkeypatch.setattr(numsim_api, "transpile", lambda kernel, *, precision: kernel)

    with pytest.raises(numsim.NumSimExecutionError, match="must not be empty"):
        numsim.run_case(case, engine=engine)

    assert not engine.executed


def test_run_case_freezes_bindings_before_mutating_reference(monkeypatch):
    source = np.array([3], dtype=np.int32)
    output = np.zeros(1, dtype=np.int32)

    def mutating_reference():
        source[0] = 99
        output[0] = 7
        return {"output": output}

    case = numsim.NumSimCase(
        kernel=object(),
        args={"source": source, "output": output},
        outputs=("output",),
        reference=mutating_reference,
    )
    monkeypatch.setattr(numsim_api, "transpile", lambda kernel, *, precision: kernel)
    engine = _FrozenNoOpEngine(source)

    report = numsim.run_case(case, engine=engine)

    assert not report.ok
    assert report.mismatches[0].actual == 0
    assert report.mismatches[0].expected == 7
    np.testing.assert_array_equal(engine.observed_source, np.array([3], dtype=np.int32))
    np.testing.assert_array_equal(source, np.array([3], dtype=np.int32))
    np.testing.assert_array_equal(output, np.array([0], dtype=np.int32))
