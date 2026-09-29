# TIRx-lite

```{container} component-subtitle
Kernel language & examples
```

```{container} lead
A PTX-level subset of the TIRx foundation IR, with a traced Python authoring API.
```

## Key idea

**TIRx is the foundation IR; TIRx-lite is a subset of it for kernel programming.**
A TIRx-lite kernel is represented as a TIRx `PrimFunc` and uses the TIRx
compiler and analysis infrastructure. The traced Python DSL is the authoring
interface to this subset.

Its defining feature is **PTX-level abstraction**: instruction operations,
operands, memory addresses, synchronization, and execution roles are explicit
in the kernel source. This gives an agent concrete operations to edit and
inspect when improving a kernel.

## What the subset includes

| Part | Included in TIRx-lite |
| --- | --- |
| Values and local storage | Typed scalar expressions, arithmetic, casts, and register-local scalars and arrays. |
| Control flow | Loops, conditional branches, and instruction predication. |
| Memory and addressing | Typed global pointers, shared-memory allocation, address calculation, and instruction operands such as tensor maps and shared-memory descriptors. |
| Instruction operations | PTX operations for memory movement, computation, synchronization, and matrix instructions, plus supported CUDA intrinsics. |
| Execution structure | Kernel launch configuration, CTA/warp/lane/thread coordinates, and explicit warp roles. |

The authoring API also supplies barrier and pipeline helpers, stage/phase
cursors, and reusable instruction sequences. These helpers compose the
low-level operations in the subset.

The subset excludes high-level tile primitives and general tensor-layout
APIs. Ordinary tensors use the default layout; global and shared-memory data
accesses use explicit instructions. Shared-memory swizzles remain explicit
allocation options.

## Core API

Import the Python interface as `tirx_kernels.tirx_lite`, conventionally named
`txl`. Decorating a function with `@txl.kernel(...)` traces it and returns a
`Kernel` object.

| Interface | Role |
| --- | --- |
| `txl.kernel(...)` | Define a kernel entry and its launch configuration. |
| `txl.gptr(dtype)`, `txl.TensorMap`, scalar dtype annotations | Describe pointer, tensor-map, and scalar parameters. |
| `txl.local_scalar(...)`, `txl.alloc_local(...)`, `txl.assign(...)` | Allocate and update register-local values. |
| `txl.If(...)`, `txl.While(...)`, `txl.serial(...)`, `txl.unroll(...)` | Build IR branches and loops through context managers. |
| `txl.cta_id(...)`, `txl.warp_id()`, `txl.lane_id()`, `txl.thread_id()` | Obtain execution coordinates for addressing and control flow. |
| `txl.smem_pool().alloc(...)` | Allocate shared-memory storage with an explicit shape, dtype, and optional swizzle. |
| `txl.specialize()` | Define named warp roles and their execution scopes. |
| `txl.MBarrier`, `txl.TMABar`, `txl.TCGen05Bar` | Construct barrier helpers. |
| `txl.Pipeline`, `txl.PipelineState`, `txl.RingState` | Construct pipeline protocols and stage/phase state. |
| `txl.ptx`, `txl.cuda`, `txl.idioms` | Spell instructions and CUDA intrinsics, or compose reusable instruction sequences. |
| `kernel.func` / `kernel.mod` | Inspect the pre-lowering TIRx `PrimFunc` or its containing `IRModule`. |
| `kernel.compile(target=None)` | Compile through the TIRx pipeline. |

The {kernels}`authoring guide <tirx_kernels/tirx_lite/README.md>` owns the full
language contract. For exact signatures and supported forms, use the
{kernels}`entry API source <tirx_kernels/tirx_lite/entry.py>` and the
{kernels}`public namespace <tirx_kernels/tirx_lite/__init__.py>` matching your
installed revision.

## Use it on its own

Install the [Python environment](../installation.md#install-python-packages).
TIRx-lite is distributed in `tirx-kernels`; importing it does not require an
agent or kcoral.

Save this as `zero.py`. It constructs a kernel that writes one float per lane
to a 32-element output buffer:

```python
import tirx_kernels.tirx_lite as txl


@txl.kernel(warps=1, arch="sm_100a", grid=1)
def zero(out: txl.gptr(txl.f32)):
    txl.ptx.st.global_.f32(out.ptr_to([txl.lane_id()]), txl.float32(0))


print(zero.func)
```

```bash
python zero.py
```

The result is a printed TIRx function, not a GPU launch. Compiling with
`zero.compile()` additionally needs a compatible CUDA toolchain; running the
result needs a GPU supporting the kernel's target.

```{note}
Keep kernel definitions in a Python file and use live type annotations.
Do not enable `from __future__ import annotations` in a TIRx-lite module.
```

## Kernel Zoo

The {kernels}`tirx-kernels collection <README.md>` provides complete kernels
with input preparation, correctness checks, and benchmark entry points.
Use it to find a nearby implementation before writing a new kernel.

| Start with | Examples |
| --- | --- |
| Basic kernels | {kernels}`FP16/BF16 GEMM <tirx_kernels/gemm/fp16_bf16_gemm.py>` and {kernels}`RMSNorm <tirx_kernels/norm/rmsnorm.py>`. |
| Attention and library ports | {kernels}`FlashAttention <tirx_kernels/ported/flashattention>`, {kernels}`FlashInfer <tirx_kernels/ported/flashinfer>`, and {kernels}`cuDNN <tirx_kernels/ported/cudnn>`. |
| Native kernels by workload | {kernels}`KDA <tirx_kernels/kda>`, {kernels}`MSA <tirx_kernels/msa>`, and the {kernels}`native kernel catalog <README.md#native-tirx>`. |

Install the dependencies used when importing the kernel examples, then list
the kernels and configurations available in your installed package:

```bash
python -m pip install tirx-harness torch pytest \
  --extra-index-url https://download.pytorch.org/whl/cu132
python -m tirx_kernels.registry --format json
```

For example, run a correctness check of the GEMM implementation:

```python
from tirx_kernels.gemm import fp16_bf16_gemm

fp16_bf16_gemm.run_test(dtype="fp16", M=1024, N=1024, K=1024)
```

This compiles and runs on a GPU supported by that kernel. Check its
`KERNEL_META` for supported architectures and its source for dependencies and
configuration choices. The package's catalog and benchmark results describe
the implementations at that revision.

To add a kernel produced by your agent, follow
[Contribute a kernel](../development/contribute-kernel.md).

## Connect it to your agent loop

Use `.func` with [Synccheck and Racecheck](tools.md#checker-api). Bind every
runtime argument to concrete inputs for the invocation you want to inspect.
Compile the kernel separately and give its launch callable to your test or
[benchmark adapter](../development/add-workload.md#define-the-contract).

Keep compile and setup work outside the timed callable when the task measures
GPU execution latency. Each benchmark owns its own candidate interface; a
`Kernel` object alone does not define the workload's input or output contract.

The [wiki skill](../installation.md#install-agent-skills) helps an agent
find the nearest implementation in the wiki's fetched `references/repos/tirx-kernels/`
checkout. Verify runtime APIs against the installed package.

## Use another kernel language

Connect your language's compiler and launch interface to the workload. You can
keep compiler analysis if the resulting TIRx uses supported operations; another IR
needs corresponding analysis tools. Supply examples for the new language.

Keep the workload's input domain, correctness policy, and scoring rule fixed
while changing components. For packaged tasks, the task declaration also
defines the permitted kernel language. To define a different contract, see
[Add a workload](../development/add-workload.md).
