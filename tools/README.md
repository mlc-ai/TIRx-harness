# Compiler analysis

TIRx Harness provides inspection, numerical simulation, and correctness checks
through compiler analysis:

```bash
python -m pip install tirx-harness
```

Supports Python 3.12/3.13 on Linux x86_64/aarch64 with glibc 2.28 or newer.
Simulation and correctness checks require Rust 1.89 or newer and a C linker.

The package exposes:

- `tirx_harness.synccheck` and `tirx_harness.racecheck`: native correctness checks
- `tirx_harness.numsim`: transpilation and deterministic numerical simulation
- `tirx_harness.dump_kernel`: CUDA, PTX, cubin, and SASS inspection

See the [installation guide](../docs/installation.md) for the full harness setup.

GPU compilation requires the CUDA toolkit. Full GPU tests also require a TVM
build with [the boolean BitwiseNot fix](https://github.com/apache/tvm/pull/20445).
