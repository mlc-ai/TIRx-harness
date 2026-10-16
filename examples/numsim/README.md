# FP8 sparse MLA: quantization versus an algorithm bug

[`fp8_sparse_mla.py`](fp8_sparse_mla.py) implements the numerical path of
DeepSeek-V4 sparse multi-head latent attention: sparse KV gather, masked and
variable-length rows, attention sinks, blockwise online softmax, and PV
accumulation. Each block scales its unnormalized softmax weights by 448, then
stores them in FP8 E4M3 before PV. The denominator and accumulators remain FP32;
the common scale cancels at final normalization.

The source algorithm is the optimized SM100a inference kernel
`tirx_kernels.mla.mla_dsv4_multishape`, validated here with `tirx-kernels==0.1.2.post1`.
That implementation covers 94 inference shapes and uses TCGEN/TMA, cluster
split-K, and BF16 output/partial storage. This example is a synchronous typed
TIRx implementation with one warp per query/head and FP32 output, so NumSim's
high mode can execute it. It also updates each block maximum eagerly, while the
production kernel can defer small updates. It does not reproduce the optimized
kernel's instruction stream or scheduling. With `--gpu`, the script separately
launches the original production implementation through its own planner on the
same inputs.

## Run

Use the repository's configured environment, including compatible TIRx TVM,
Rust, and `ml-dtypes`. From the repository root:

```bash
python examples/numsim/fp8_sparse_mla.py
python examples/numsim/fp8_sparse_mla.py --gpu
```

The first command only requires CPU execution. `--gpu` also needs PyTorch CUDA,
B200/SM100a, and the installed `tirx-kernels` package. The production adapter
uses that package's private `_candidate_setup` entry point; the version appears
in the JSON report. Its supported example shapes require head dimension 512,
8/16/32/64/128 heads, and at least 128 selected slots.

The default shape is 4 queries, 8 heads, head dimension 512, and 256 selected KV
slots per query, processed in blocks of 64. Inputs include nonidentity sparse
indices, invalid slots, varying lengths, finite sinks, and an empty final query.
`--selected 1024` exercises a longer sparse context. Other options are
`--queries`, `--heads`, `--dimension`, `--seed`, and `--workers`.

## What changes between modes

All modes start from **identical, already FP8-quantized Q and KV inputs**. No
mode recovers their lost input bits. The independent mathematical reference
promotes those values to FP64 and evaluates dense attention with a stable
softmax, without the kernel's online recurrence.

- `fp8`: rounds the intermediate probability weights to E4M3; QK, softmax
  statistics, and PV accumulation use FP32.
- `fp32`: changes the probability storage to FP32. Q/KV storage remains FP8.
- `high`: executes the original FP8-source example with `precision="high"`,
  retaining FP64 intermediate values and output in NumSim shadow storage.
- `wrong_fp8` / `wrong_high`: multiply the final output by an erroneous 1.01.

A second NumPy reference explicitly rounds each block's probabilities to E4M3
using `ml_dtypes`. The native FP8 result must agree with this quantized oracle
at `rtol=2e-5, atol=2e-6`. With `--gpu`, the example's GPU and NumSim outputs are
compared directly, and both are checked against the corresponding oracle.
These are tolerance comparisons, not GPU bitwise-equality claims.

## Observed errors

Default shape, seed 7, against the mathematical FP64 reference:

| Implementation / mode | Maximum absolute error | Relative L2 error |
| --- | ---: | ---: |
| Example, FP8 probabilities | 8.510e-3 | 2.225% |
| Example, FP32 probabilities | 2.571e-7 | 2.923e-7 |
| Example, high / FP64 | 6.106e-16 | 7.604e-16 |
| Example, wrong FP8 | 9.443e-3 | 2.442% |
| Example, wrong high | 3.479e-3 | 1.000% |
| Original production FP8 kernel, B200 | 8.727e-3 | 2.149% |

The example's FP8 result differs from its quantized oracle by at most `5.26e-8`.
GPU-versus-NumSim maximum differences are `5.96e-8` for FP8 probabilities and
`1.04e-7` for FP32 probabilities. The original production result passes its
published FP8 tolerances (`atol=0.02, rtol=0.06`) against the independent FP64
reference; its native output is BF16.

An **illustrative 5% relative-L2 gate** accepts both the correct FP8 result and
the 1% output-scale bug. This gate is distinct from the production tolerance.
At `rtol=1e-5, atol=1e-6`, the correct FP32 and high results pass while the
wrong high result fails. High additionally passes `rtol=atol=1e-10`. These
observations illustrate why a loose native comparison alone can miss a bug;
they are measured errors for these inputs, not worst-case error bounds or a
proof of kernel correctness for all inputs.

A longer context also passes the oracle and GPU checks:

```bash
python examples/numsim/fp8_sparse_mla.py --selected 1024 --seed 19 --gpu
```

With the same query/head dimensions and 16 online-softmax blocks per row:

| Implementation / mode | Maximum absolute error | Relative L2 error |
| --- | ---: | ---: |
| Example, FP8 probabilities | 4.134e-3 | 2.046% |
| Example, FP32 probabilities | 2.998e-7 | 5.356e-7 |
| Example, high / FP64 | 6.661e-16 | 1.430e-15 |
| Original production FP8 kernel, B200 | 4.173e-3 | 1.908% |

The wrong FP8 result still passes the illustrative 5% gate (2.264% relative L2),
while wrong high fails the strict comparison (1.000% relative L2).

On 2026-10-11, a 224-CPU host with four engine workers took 0.400 s for native
FP8, 0.403 s for native FP32, and 10.067 s for high mode. Each is one
`Engine.run` measurement, excluding compilation and references. Preflight
recorded 94.78% CPU idle, load averages 15.30/16.60/16.00, and CPU pressure
`some avg10=0.00`. These CPU simulation times do not measure GPU latency.
