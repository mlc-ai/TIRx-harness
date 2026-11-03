# Dump Kernel Source

Extract the generated CUDA source from a compiled TIRx module and derive PTX,
cubin, and SASS. The installed `tirx_harness.dump_kernel` API returns artifact
text plus the `ptxas -v` register/spill/SMEM report.

## Python API

Call `dump_module()` on the `tvm_tirx.compile` output (the `Executable`, or a
runtime `Module`):

```python
from tirx_harness.dump_kernel import dump_cuda, dump_module

ex = tvm_tirx.compile(tvm_tirx.IRModule({"main": pf}), target="cuda", pipeline="tirx")

d = dump_module(ex, ptx=True)         # -> DumpResult
assert d.ok, d.errors                 # d.ok is False if a requested stage failed
d.cuda, d.ptx, d.sass                 # artifact text (ptx/sass only when requested)
d.ptxas                               # {'registers': 153, 'spill_stores': 0, 'barriers': 16, ...}
d.symbols                             # ['_kernel_kernel']

dump_module(ex, outdir=artifact_dir, name="kernel")  # persist files at a caller-provided path
cuda = dump_cuda(ex)                  # just the CUDA source string
```

If a requested stage fails (bad `arch`, unknown `function`, …) its field stays
`None`, `d.ok` is `False`, and a message is appended to `d.errors` (printed to
stderr only when `quiet=False`).

### Options (`dump_module`)

- `sass=True` (default), `ptx=False` — which artifacts to build (CUDA is always returned).
- `arch=None` — nvcc/ptxas arch; **auto-detected** from the local GPU (B200 →
  `sm_100a`). The architecture-specific `a` suffix (needed for `tcgen05.*` on
  Blackwell / Hopper-only opcodes; ptxas rejects them without it) is added
  automatically for SM 9.x/10.x. Override with e.g. `arch="sm_90a"`.
- `function=None` — scope SASS to one kernel symbol (multi-kernel cubins).
- `lineinfo=True` — interleave CUDA source line numbers into SASS (maps a SASS
  instruction back to its CUDA origin when investigating spills/scheduling).
- `outdir=None`, `name="kernel"` — when `outdir` is set, artifacts are written as
  `<name>.{cu,ptx,cubin,sass,ptxas.log}` and their paths returned in `.paths`;
  otherwise a temp dir is used and cleaned up.

`dump_module` accepts either the `Executable` from `tvm_tirx.compile` or a runtime
`Module`; `dump_cuda` extracts just the CUDA source (walks the imported-module
tree — the host module alone doesn't carry the device source).

## Notes

- **Kernel symbol name** — the device symbol is `<global_symbol>_kernel`, where
  `global_symbol` comes from the **PrimFunc's own attribute**, not the IRModule
  key. TIRx kernels carry `global_symbol = "_kernel"`, so `IRModule({"main":
  kernel})` compiles to `_kernel_kernel` (*not* `main_kernel`). `dump_module`
  returns it in `.symbols`; pass one to `function=` to scope SASS.
- **CUDA** shows the generated C++ kernel source — verify codegen, register
  annotations, high-level structure.
- **PTX** shows the virtual ISA — instruction selection, register allocation, predication.
- **SASS** shows the actual machine code — count instructions, verify scheduling,
  check for spills. The `ptxas -v` report (`.ptxas`) is essential for catching
  spill regressions. It has no `smem` key for kernels using `extern __shared__`
  dynamic SMEM (most TIRx GEMM/attention kernels) — ptxas only reports *static*
  SMEM, so an absent `smem` ≠ "0 bytes SMEM".
