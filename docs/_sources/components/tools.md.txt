# Compiler analysis

```{container} component-subtitle
Domain-specific compiler analysis
```

```{container} lead
Inspect numerical behavior, synchronization, and memory races directly from TIRx.
```

## Key idea

TIRx Harness provides compiler analysis through the
`tirx_harness` Python package. Three CPU tools share the same TIRx execution model:
**NumSim** enables numerical iteration without GPU access, **Synccheck** checks
synchronization before GPU execution, and **Racecheck** checks memory-access
ordering. Each consumes a
kernel and concrete inputs, so an agent can investigate a candidate before
running it on a GPU. The package also exports generated-code inspection.

NumSim, Synccheck, and Racecheck run on the CPU. See the
[installation guide](../installation.md#install-python-packages) to get started.

## NumSim

**When to use it.** Use NumSim when getting a real GPU run is costly or
inconvenient:

- **Shared or scarce GPUs:** keep iterating while a job waits in the queue or
  the target GPU is occupied.
- **Multi-GPU workloads:** investigate a supported individual kernel's
  computation before reserving the devices and launching the full distributed
  job.
- **CPU-only development and CI:** run supported numerical regression cases
  on a development machine or CI worker without assigning a GPU to every edit.
- **Many agent-generated candidates:** screen small cases on CPU before
  submitting candidates to a remote or metered GPU service.

NumSim moves numerical debugging into the local development loop. With small,
representative inputs and an independent reference, it can identify output
mismatches by index and value before you spend a GPU allocation on the
candidate. Resolve synchronization and race findings first; use the target
GPU to validate selected candidates and measure performance.

**Mechanism.** NumSim transpiles a specialized TIRx kernel into a cached native
Rust artifact. Its CPU engine executes concrete control flow and modeled
instructions, maintaining register values, memory, and asynchronous-operation
state. It returns output arrays and diagnostics; comparison against an
independent reference identifies numerical mismatches.

**Runnable example.** Save this kernel as `vector_add.py`:

```python
import tirx_kernels.tirx_lite as txl


@txl.kernel(warps=1, arch="sm_100a", grid=1)
def kernel(a: txl.gptr(txl.f32), b: txl.gptr(txl.f32), out: txl.gptr(txl.f32)):
    lane = txl.lane_id()
    x = txl.local_scalar("float32")
    y = txl.local_scalar("float32")
    txl.ptx.ld.global_.f32(x, a.ptr_to([lane]))
    txl.ptx.ld.global_.f32(y, b.ptr_to([lane]))
    txl.ptx.st.global_.f32(out.ptr_to([lane]), x + y)
```

**API.** Import `numsim` from `tirx_harness`. For a tirx-lite vector-add kernel
with three 32-element `float32` buffers named `a`, `b`, and `out`, bind concrete
NumPy arrays and compare the result with a NumPy reference:

```python
import numpy as np
from tirx_harness import numsim
from vector_add import kernel

a = np.arange(32, dtype=np.float32)
b = np.ones(32, dtype=np.float32)
expected = a + b

module = numsim.transpile(kernel.func)
result = numsim.Engine().run(
    module,
    inputs={"a": a, "b": b, "out": np.zeros_like(a)},
    outputs=("out",),
)
np.testing.assert_allclose(
    result.outputs["out"], expected, rtol=1e-5, atol=1e-8, equal_nan=False
)
```

Save this as `check_numsim.py` beside `vector_add.py` and run
`python check_numsim.py`.

Dictionary keys must match kernel parameter names; include output buffers
alongside input arrays and any scalar parameters. Choose
tolerances according to the workload; use `np.testing.assert_array_equal`
when exact equality is required. Decode buffers carrying encoded values
before comparing their numerical contents.

| Interface | Main parameters | Result |
| --- | --- | --- |
| `numsim.transpile(func, *, cache_dir=None)` | `func`: specialized TIRx function. `cache_dir`: optional artifact-cache directory. | `CompiledModule` |
| `numsim.Engine(...)` | `max_workers=8`: positive CPU-worker count, or `"auto"` to use the detected CPU count. | Execution engine |
| `engine.run(module, inputs, *, outputs=None)` | `module`: transpiled artifact. `inputs`: concrete binding dictionary. `outputs`: buffer names or a mapping from result names to buffer names; `None` selects bound output buffers. | `NumSimResult` |

`NumSimResult` exposes `.outputs`, `.diagnostics`, and `.stats`. Keep simulator
diagnostics alongside the workload's numerical comparison result.

See the {repo}`NumSim API source <tirx_harness/src/tirx_harness/numsim/api.py>`
for full signatures.

**Limitations.**

- Only modeled TIRx operations and their supported dtype, shape, and modifier
  combinations can execute. Opaque CUDA bodies are unsupported; consult the
  {repo}`operation coverage table <tirx_harness/src/tirx_harness/numsim/engine-rs/SUPPORTED_OPS.md>`.
- Hardware timing and some instruction results use deterministic
  representatives. Simulation time is not GPU latency, and numerical fidelity
  depends on the operation's documented model.
- NumSim is not a numerical oracle or a race/synchronization verifier. Use
  independent reference outputs, the two checkers below, and device tests.

## Synccheck

**When to use it.** Run Synccheck as a CPU precheck before submitting a
candidate to the GPU. Repeat it after kernel changes, especially changes to
barrier arrivals and waits, pipeline stage reuse, or warp roles. Its findings
help you fix synchronization protocol errors before spending GPU time on a
launch that could hang.

**Guarantee.** Within its supported model, Synccheck detects:

- Barrier use without initialization ordered before it, or reuse without the
  required wait/consumption dependencies.
- Invalid barrier participants, arrival counts, or asynchronous completion
  counts.
- Synchronization deadlocks and inconsistent final protocol states.

The check covers **all interleavings allowed by program order and
synchronization dependencies**, with each warp's executed path and values
held fixed. It therefore detects synchronization errors that another
warp/completion order can expose, even when the CPU simulation happens to
finish successfully.

**Mechanism.** The algorithm has three steps:

1. **Fully simulate the kernel.** Execute supported computations, memory
   accesses, and data-dependent branches and loops on CPU. Record a separate
   operation sequence for each warp, including its synchronization and
   asynchronous events.
2. **Check dependencies and counts.** Retain the order within each warp and
   cross-warp dependencies established by synchronization. For common barrier
   patterns, check each phase's
   arrival/completion counts and required dependency chains, such as
   initialization before use.
3. **Explore the remaining patterns.** Keep each warp's next operation,
   blocked status, barrier state, and pending completions. Try the next events
   permitted by those dependencies, including orders different from the CPU
   run; branch when several events are possible and merge equivalent states.
   Report protocol violations or unfinished
   states where no event can progress.

For example, warp A initializes a barrier and warp B uses it. The checker
requires a dependency chain guaranteeing that A's initialization precedes
B's use. A merely running first in the CPU simulation does not establish
that guarantee.

**API:** `synccheck(kernel, inputs=None)` returns a `SyncCheckReport`.

Bind NumPy arrays by kernel parameter name, including output buffers. For
example, for three 32-element `float32` buffers named `a`, `b`, and `out`:

```python
import numpy as np
from tirx_harness import synccheck
from vector_add import kernel

report = synccheck(
    kernel.func,
    inputs={
        "a": np.arange(32, dtype=np.float32),
        "b": np.ones(32, dtype=np.float32),
        "out": np.zeros(32, dtype=np.float32),
    },
)
report.print()
report.require_clean()
```

Save this as `check_synccheck.py` beside `vector_add.py` and run
`python check_synccheck.py`.

Both arguments and the report interface are described in
[Checker API](#checker-api).

**Limitations.** Data-dependent execution is supported, but the verification
covers the synchronization program selected by this invocation. Alternate
inputs, ordinary-memory values, atomic return orders, and the different
control-flow paths they might select are not enumerated.

## Racecheck

**When to use it.** Run Racecheck as a CPU precheck before submitting a
candidate to the GPU. Repeat it after changing memory-access patterns,
buffer reuse, or asynchronous transfers. It helps catch missing synchronization
between memory accesses before you spend GPU time on results that may depend
on execution order.

**Guarantee.** Within its supported model, Racecheck detects:

- Read/write and write/write conflicts in global memory, shared memory, or
  TMEM that lack the required ordering, including accesses through aliases of
  the same storage.
- Missing memory ordering, such as a required release/acquire dependency or
  proxy fence between ordinary and asynchronous memory accesses, even when
  their execution order is established.
- Accesses outside a buffer view or its backing allocation.
- Reuse of memory still accessed by an unfinished asynchronous operation.

For the accesses selected by this invocation, the check requires both
**execution ordering and the necessary memory-ordering dependencies**. It can
detect a race even when the CPU simulation produces the expected output.

**Mechanism.** The algorithm has three steps:

1. **Fully simulate the kernel.** Execute supported computations, memory
   accesses, and data-dependent branches and loops on CPU. Record each active
   lane's physical byte ranges, reads and writes, and synchronization events.
2. **Track ordering dependencies.** Use vector clocks: compact records of
   which earlier accesses each lane or asynchronous operation is ordered
   after. Update these records through modeled program order, barriers,
   release/acquire operations, and asynchronous completion waits. Also track
   required memory and proxy fences to check visibility between accesses.
3. **Check overlapping accesses.** For ordinary reads and writes, report a
   race when two accesses overlap, at least one writes, and neither must
   happen before the other. Compare physical bytes so different buffer names
   cannot hide an overlap. Atomic accesses follow the model's atomicity and
   scope rules.

For example, warp A writes a shared-memory value and warp B reads it. The
checker requires a dependency chain ordering the write before the read.
A merely writing first in the CPU simulation does not establish that
guarantee.

**API:** `racecheck(kernel, inputs=None)` returns a `RaceReport`.

Bind NumPy arrays by kernel parameter name, including output buffers. For
example, for three 32-element `float32` buffers named `a`, `b`, and `out`:

```python
import numpy as np
from tirx_harness import racecheck
from vector_add import kernel

report = racecheck(
    kernel.func,
    inputs={
        "a": np.arange(32, dtype=np.float32),
        "b": np.ones(32, dtype=np.float32),
        "out": np.zeros(32, dtype=np.float32),
    },
)
report.print()
report.require_clean()
```

Save this as `check_racecheck.py` beside `vector_add.py` and run
`python check_racecheck.py`.

Both arguments and the report interface are described in
[Checker API](#checker-api).

**Limitations.** Data-dependent execution is supported, but the check covers
the accesses selected by this invocation. Alternate inputs, ordinary-memory
values, atomic return orders, and the different control-flow paths or addresses
they might select are not enumerated.

## Checker API

Synccheck and Racecheck share these public parameters:

| Parameter | Meaning |
| --- | --- |
| `kernel` | A TIRx `PrimFunc`; pass `.func` from a tirx-lite `Kernel`. |
| `inputs=None` | Dictionary from parameter names to concrete scalars and CPU NumPy buffers, including output storage. Parameterless kernels may omit it; otherwise supply all runtime bindings. |

For tensor-map bindings, `tirx_harness.numsim.TensorMap(...).numpy()` constructs
the simulator's descriptor array. The
{repo}`binding types <tirx_harness/src/tirx_harness/numsim/cases.py>` define its
shape, strides, dtype, and swizzle parameters.

| Report interface | Meaning |
| --- | --- |
| `.verdict` | `clean`, `review`, `incomplete`, or `error`. |
| `.findings` | Structured findings with status, kind, message, and source/witness evidence. |
| `.to_dict()` | JSON-safe report for an agent or artifact store. |
| `.print()` | Human-readable findings and available source context. |
| `.require_clean()` | Raise unless the verdict is `clean`. |

A verdict covers the supplied specialization, launch, and inputs. The public
checkers currently require a single kernel phase. Missing bindings,
unsupported effects, and coverage limits produce `incomplete`; `review`
indicates an advisory, and `error` indicates a detected violation. A clean
report does not establish correctness for other inputs or replace an
independent GPU correctness test. The
{repo}`checker entry points <tirx_harness/src/tirx_harness/numsim/checkers.py>`
own these signatures.

## Inspecting generated code

Use generated-code inspection when an edit changes performance unexpectedly,
or to confirm the instructions and register, spill, and shared-memory usage
produced by compilation.

`tirx_harness.dump_kernel.dump_module` extracts CUDA, PTX, SASS, and compiler
resource information from a compiled module:

```python
from tirx_harness.dump_kernel import dump_module


def inspect_candidate(compiled_module, artifact_dir):
    result = dump_module(
        compiled_module, ptx=True, outdir=artifact_dir, name="candidate"
    )
    if not result.ok:
        raise RuntimeError(result.errors)
    return result.paths
```

Save both code blocks in `inspect_vector_add.py` beside `vector_add.py`,
then run `python inspect_vector_add.py`:

```python
from vector_add import kernel
from tvm.target import Target

target = Target({"kind": "cuda", "arch": "sm_100a"})
with target:
    compiled = kernel.compile(target=target)
print(inspect_candidate(compiled, "artifacts/vector-add"))
```

`ptx` and `sass` select artifact stages; CUDA source is always returned.
`outdir` and `name` control saved files. Requested stages
need their CUDA-toolkit executables and can fail independently. See the
{repo}`source-dump API <skills/tirx-profile-kernel/references/dump-source.md>`.

## External tools

TIRx Harness uses these external tools in its agent loop for GPU-side
debugging and profiling. The [debugging and profiling skills](../installation.md#install-agent-skills)
guide the agent to select a tool, capture its reports or traces, and use that
evidence to decide how to revise the kernel.

| Tool | Role in the agent loop |
| --- | --- |
| {repo}`Nsight Compute <skills/tirx-profile-kernel/references/ncu.md>` | The agent uses hardware-counter reports to identify stalls, utilization limits, and memory bottlenecks before choosing an optimization. |
| {repo}`IKET <skills/tirx-profile-kernel/references/iket.md>` | The agent uses annotated timelines to locate pipeline bubbles, missing overlap, and load imbalance. |
| {repo}`Compute Sanitizer <skills/tirx-debug-kernel/references/compute-sanitizer.md>` | The agent uses device-side error reports to investigate memory, initialization, race, or synchronization failures in the launched binary. |

The agent runs these tools on a local GPU or through the harness's
[kcoral adapters](kcoral.md#framework-adapters) for remote execution and
artifact retrieval. After revising the kernel, it reruns correctness checks
and the ordinary benchmark. Profiler replay and instrumentation timings
serve diagnosis; benchmark results determine performance improvements.

## Replace analysis checks

Add or replace checks for the properties you need, and return their results to
the agent. Keep the workload's independent numerical reference.
