# TIRx Harness

```{container} lead
Give your agent the language, tools, and execution environment to develop
kernels for graphics processing units (GPUs). Use the components directly or
connect them in an agent loop.
```

::::{container} hero-actions
:::{button-ref} quick-start
:color: primary
Get started
:::
:::{button-link} https://github.com/mlc-ai/TIRx-harness
:color: secondary
:outline:
View on GitHub
:::
::::

TIRx Harness brings together **TIRx-lite** for kernel authoring,
**compiler analysis** for kernel inspection, and [**kcoral**](https://kcoral.mlc.ai/)
for remote execution. Skills guide the agent in using them, while workload
contracts define correctness and performance. Start with the component or task you need.

## How the pieces fit

```{container} architecture
**Your agent orchestrates the loop.**

Skills guide its choices. TIRx-lite produces a kernel; compiler analysis inspects
its TIRx function. A benchmark runs the compiled candidate locally or through
kcoral. Correctness results, timings, and profiler artifacts inform the next edit.
```

## Choose a component

:::::{grid} 1 1 2 3
:gutter: 3
:class-container: component-grid

::::{grid-item-card} TIRx-lite
:link: components/TIRx-lite
:link-type: doc

:::{container} component-subtitle
Kernel language & examples
:::

Write kernels in TIRx-lite, a domain-specific language over the TIRx
intermediate representation. Explore complete implementations and launch examples.
+++
Language & examples →
::::

::::{grid-item-card} Compiler analysis
:link: components/tools
:link-type: doc

:::{container} component-subtitle
Domain-specific compiler analysis
:::

Check synchronization, memory races, and numerical behavior.
Inspect compiler output and generated GPU instructions.
+++
Tools & APIs →
::::

::::{grid-item-card} kcoral
:link: https://kcoral.mlc.ai/
:link-type: url

:::{container} component-subtitle
Remote GPU execution
:::

Keep your agent on one machine and run GPU work on another.
Use remote execution adapters to run kernels and retrieve diagnostic artifacts.
+++
Project website →
::::

:::::

## Choose your next step

| What you want to do | Start here |
| --- | --- |
| Optimize a kernel in your own project | [Quick Start](quick-start.md) |
| Run a registered workload | [Optimization Runs](optimization-runs.md) |
| Define a new optimization task | [Add a workload](development/add-workload.md) |
| Publish a kernel produced by a run | [Contribute a kernel](development/contribute-kernel.md) |
| Reproduce and resolve a defect | [Report and fix bugs](development/report-and-fix-bugs.md) |

For hardware and kernel-programming background, see
[Modern GPU Programming for MLSys](https://mlc.ai/modern-gpu-programming-for-mlsys/).
For complete implementations, browse the
[Kernel Zoo](components/TIRx-lite.md#kernel-zoo).

```{toctree}
:hidden:
:maxdepth: 1
:caption: Get Started

Overview <self>
installation
quick-start
optimization-runs
```

```{toctree}
:hidden:
:maxdepth: 1
:caption: Components

Kernel language & examples <components/TIRx-lite>
Compiler analysis <components/tools>
Remote GPU execution <components/kcoral>
```

```{toctree}
:hidden:
:maxdepth: 1
:caption: Development

development/add-workload
development/contribute-kernel
development/report-and-fix-bugs
development/build-the-docs
```
