# NumSim Tests

These source-tree-only tests are not packaged in the `tirx-harness` wheel.

## Structure

- `abi/`: unified artifact compatibility version and public Rust boundary.
- `registry/`: public-op discovery, direct registry emission, specialization
  invariance, and generated support-matrix checks.
- `runtime/`: focused numerical, bit-level, layout, lane, and state oracles.
- `integration/`: end-to-end analysis, Rust generation, build/load, bindings,
  execution, mismatch reporting, source maps, and caches.
- `microtests/`: the same small PrimFunc and inputs compared between NumSim and
  a live GPU.
- `corpus/`: representative complete kernels. Runnable complete-kernel tests
  compare GPU, NumSim, and an independent reference pairwise.
- `support/`: reusable kernels, runtime input domains, and handwritten
  operation-to-test assignments. Production NumSim code must not import it.
- `conftest.py`: shared fixtures and the GPU opt-out flag.

## Operation Specializations

An accepted canonical call or tile op is specialized only by transpile-time
facts, which select the engine instruction and variant its site emits
(`support/manifest.py` `emitted_calls`). Runtime values remain ordinary call
operands. Specialization tests assert on that emitted text; there is no
separate specialization key.

Production registries are the support source of truth. The test-owned
`support/runtime_cases.py` table only points each registered operation at one
or more handwritten correctness tests. Registry tests check that every
registered operation is represented and every referenced test exists. One
operation may have several runtime cases when more numerical or state variants
are useful.

There is no form census, equivalence layer, runtime witness accounting, or
session-wide conformance gate. Runtime tests directly assert outputs, physical
bits, layout effects, or engine state.

## Confidence Path

```text
public TIRx API
    -> canonical registry and emitted engine instruction
    -> focused runtime/integration oracle
    -> NumSim versus GPU microtest
    -> representative kernel corpus
```

Put registry ownership and dispatch checks in `registry/`, focused semantics in
`runtime/`, cross-boundary behavior in `integration/`, GPU comparisons in
`microtests/`, and complete kernels in `corpus/`. Keep NumSim-only helpers out
of the installed `tirx_kernels` package. Fix canonical kernels in their own
repository when the kernel itself is wrong.

When hardware behavior is uncertain, add the smallest paired NumSim/GPU
microtest that varies one factor at a time and asserts concrete output values.
Do not infer an instruction rule from a corpus mismatch alone. Keep the focused
experiment in `microtests/` as regression evidence after the model is fixed.

For general runnable complete-kernel corpus cases, assert all three relations:

```text
GPU == NumSim        simulator fidelity
GPU == reference     kernel algorithm correctness
NumSim == reference  direct mismatch
```

Use high parallelism and work stealing:

```bash
python -m pytest -q -n 16 --dist=worksteal tools/tests/numsim
```

GPU microtests run by default. Pass `--no-run-numsim-gpu` for a CPU-only run.
