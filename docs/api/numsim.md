# Numerical simulation

NumSim is the CPU numerical simulator exposed by `from tirx_harness import
numsim`. It compiles a specialized TIRx function into a cached Rust artifact,
executes it with concrete inputs, and lets you compare its outputs with an
independent reference. See the [runnable example and supported-model
limitations](../components/tools.md#numsim).

## Compile and execute

```{eval-rst}
.. autoapifunction:: tirx_harness.numsim.transpile
```

Pass a TIRx function, such as a TIRx-lite kernel's `.func`, and optionally a
cache directory. Keyword parameters beginning with `_` are implementation
controls; ordinary callers should leave them at their defaults.

```{eval-rst}
.. autoapiclass:: tirx_harness.numsim.CompiledModule
   :members: cache_key, rust_source, library_path
   :undoc-members:
```

Obtain a compiled module from `transpile`; pass it to `Engine.run` or the
inspection helpers below.

```{eval-rst}
.. autoapiclass:: tirx_harness.numsim.Engine
   :members: run, max_workers, native_loop_iteration_budget, native_loop_reschedule_quantum
   :undoc-members:
```

`max_workers` must be a positive integer, or `"auto"` to use the detected CPU
count. The loop budget bounds native loop iterations; the reschedule quantum
controls how often loop execution yields to other work. Both must be positive
integers.

For `run`, `inputs` maps kernel parameter names to concrete scalars and CPU
NumPy buffers, including output storage. `outputs` selects buffer names or maps
result names to buffer names; `None` selects the bound output buffers.
The [input types](inputs.md) describe optional launch subsets and assumptions.

```{eval-rst}
.. autoapiclass:: tirx_harness.numsim.NumSimResult
   :members: outputs, diagnostics, stats, verdict, assert_close
   :undoc-members:
```

`outputs` contains the selected arrays, `diagnostics` contains simulator
messages, and `stats` contains execution statistics. The result's `verdict`
reflects simulator advisories; compare against a reference to check numerical
correctness. `assert_close` performs that comparison and raises on a mismatch.
Simulation statistics do not measure GPU latency.

## Compare outputs

```{eval-rst}
.. autoapifunction:: tirx_harness.numsim.compare
```

`expected` must be a nonempty mapping from output names to reference arrays.
Use `tolerances` to supply a {py:class}`~tirx_harness.numsim.ComparisonSpec` per
output. Without an explicit specification, integer and boolean arrays use
exact equality; floating-point comparisons use the specification's defaults.

```{eval-rst}
.. autoapifunction:: tirx_harness.numsim.run_case
```

Compile, execute, and compare a {py:class}`~tirx_harness.numsim.NumSimCase` in
one call. Supply an existing engine to reuse its execution configuration.

```{eval-rst}
.. autoapiclass:: tirx_harness.numsim.report.NumSimReport
   :members: ok, mismatches, diagnostics, verdict, require_ok
   :undoc-members:
```

`require_ok()` raises `AssertionError` when a numerical comparison fails.
An advisory can leave `ok=True` while `verdict` is `review`; keep the diagnostics
alongside the numerical result.

```{eval-rst}
.. autoapiclass:: tirx_harness.numsim.report.Mismatch
   :members: output, index, actual, expected, render
   :undoc-members:
```

## Inspect the simulator artifact

```{eval-rst}
.. autoapifunction:: tirx_harness.numsim.dump_rust
```

```{eval-rst}
.. autoapifunction:: tirx_harness.numsim.dump_semantic_manifest
```

## Exceptions

These exceptions describe transpilation, build, and execution failures.
Checker entry points instead return reports for the failures they handle;
inspect their verdicts as described in [Checkers and reports](checkers.md).

```{eval-rst}
.. autoapiexception:: tirx_harness.numsim.NumSimError
   :show-inheritance:
```

```{eval-rst}
.. autoapiexception:: tirx_harness.numsim.UnsupportedTIRxError
   :show-inheritance:
```

```{eval-rst}
.. autoapiexception:: tirx_harness.numsim.UnmodeledTIRxFormError
   :show-inheritance:
```

```{eval-rst}
.. autoapiexception:: tirx_harness.numsim.NumSimBuildError
   :show-inheritance:
```

```{eval-rst}
.. autoapiexception:: tirx_harness.numsim.NumSimExecutionError
   :show-inheritance:
```

```{eval-rst}
.. autoapiexception:: tirx_harness.numsim.InvalidLaunchError
   :show-inheritance:
```
