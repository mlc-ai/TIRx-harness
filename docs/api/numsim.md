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

## High precision for algorithm checks

Use `precision="high"` to execute supported floating operations in FP64:

```python
module = numsim.transpile(kernel, precision="high")
result = numsim.Engine().run(module, inputs, outputs=("output",))
report = numsim.compare(
    result,
    {"output": reference_fp64},
    tolerances={"output": numsim.ComparisonSpec(rtol=1e-10, atol=1e-12)},
)
assert report.precision == "high"
report.require_ok()
```

`numsim.run_case(case, precision="high")` offers the same mode. Construct the
reference independently from the **same already quantized inputs**, promoted
to FP64, and retain its FP64 outputs. Choose tolerances for the algorithm's
conditioning and reduction order. A native-mode reference rounded to FP16 or
BF16 is not a high-precision reference. High outputs are decoded FP64 arrays;
do not specify `actual_encoding="bfloat16"` for them.

For kernel agents, compare both modes with independent references. A correct
formula can disagree in native mode because of cancellation, accumulated
rounding, or a simulator approximation. Agreement in high mode helps isolate
these effects. An incorrect index, omitted term, or wrong sign should still
fail; include such negative controls and varied inputs when judging a kernel.
The LayerNorm regression in
{repo}`test_high_precision_layernorm.py <tools/tests/numsim/corpus/test_high_precision_layernorm.py>`
compares a moments-based implementation with a centered-variance reference,
and verifies that an incorrect mean divisor fails in high mode.

The promoted model has these semantics:

- FP16, BF16, FP32, FP64, E4M3FN and E8M0FNU typed values use FP64 arithmetic.
  Runtime floating casts and stores retain the FP64 result. Input encodings
  and floating literals retain their original quantization; lost input bits
  cannot be recovered. Existing FP64 computations gain no extra precision.
- Allocation sizes, byte addresses, ownership, layout maps, integer arithmetic,
  masks and scheduling mechanisms retain their original representation. A
  typed FP16 store still occupies two physical bytes. A shadow value associated
  with that address carries the FP64 result through later typed loads and
  synchronous tile copies. Global shadows persist across phases of one
  compiled module and reset for each `Engine.run`.
- Scalar arithmetic, supported mathematical intrinsics, typed warp shuffles,
  warp reductions, synchronous tile copy/cast/elementwise/reduction operations,
  and FP16/BF16 `tile.gemm` warp MMA are promoted. GEMM accumulates in FP64 in
  increasing K order using FMA. Transcendentals use host FP64 math, without GPU
  calibration tables. Floating tile arithmetic drops low-precision rounding
  and FTZ behavior.
- `result.outputs` contains unrounded FP64 floating outputs; integer outputs
  keep their dtype. Caller-provided buffers keep their native size and receive
  a narrowed byte projection, which is not the result of a native execution.
  For FP16/BF16/FP8 the byte projection goes through FP32; use `result.outputs`
  for numerical validation. A separate simulation starts from the host
  buffers' projected bytes; use a compiled kernel sequence to retain shadows.
- Accesses that reinterpret promoted values through a different dtype, or use
  only some of their bytes, fail instead of interpreting the narrowed
  projection. Accessed integer cells also retain their width, so partial or
  mixed-width integer accesses are rejected. Packed floating
  arithmetic, floating bit reinterpretation, vector accesses, raw PTX numeric
  instructions, atomics, asynchronous copies, TensorMaps, TCGEN/TMEM, and
  zero-filled scalar accesses are currently unsupported in high mode.
  Unsupported operations raise `UnsupportedTIRxError`; ambiguous accesses
  discovered during execution raise `NumSimExecutionError`.

High precision does **not** establish GPU bitwise equality, acceptable native
error, numerical stability, or race/synchronization correctness. Promotion can
also change data-dependent branches. Kernels intentionally depending on
quantization must be checked in native mode against their intended semantics.
Run the native checkers and device validation separately. FP64 itself rounds,
can overflow, and can lose information in ill-conditioned computations.

Native precision remains the default. The two modes use separate compiled
artifacts. High mode adds memory proportional to accessed typed cells and
extra work for shadow lookups; its simulation time is not a GPU performance
estimate.

For example, on a 224-CPU host with four engine workers, the `128 x 1024`
LayerNorm case above had maximum absolute error `0.176` in native mode and
`3.23e-10` in high mode against the centered FP64 reference. Three `Engine.run`
measurements were `103.3 / 54.2 / 54.7 ms` native and `1.22 / 1.15 / 1.02 s`
high. The deliberately wrong mean divisor still failed in high mode, with
maximum error `2.48`.
These measurements exclude compilation and reference construction
(2026-10-10; host preflight: 90.8% CPU idle, load average 27.6/42.4/49.1).
They demonstrate error isolation for this case, not a universal error bound.

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
   :members: ok, mismatches, diagnostics, precision, verdict, require_ok
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
