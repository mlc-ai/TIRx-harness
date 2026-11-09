<div align="center" id="top">

# TIRx Harness

[![Documentation](https://img.shields.io/badge/docs-latest-green)](https://tirxharness.mlc.ai/docs/)
[![Book](https://img.shields.io/badge/book-Agentic_GPU_Programming-blue)](https://mlc.ai/agentic-gpu-programming-for-mlsys/)
[![Related Repository: tirx-kernels](https://img.shields.io/badge/Related_Repo-tirx--kernels-fafbfc?logo=github)](https://github.com/mlc-ai/tirx-kernels)

**An Open Compiler Harness for Agentic GPU Programming**

[Get Started](#get-started) | [Documentation](https://tirxharness.mlc.ai/docs/) | [Book](https://mlc.ai/agentic-gpu-programming-for-mlsys/) | [Blogpost](https://blog.mlc.ai/2026/09/29/tirx-harness-an-open-compiler-harness-for-agentic-gpu-programming)

</div>

## Overview

TIRx Harness is a compiler harness combining a minimal stable compiler
foundation, a knowledge base, tools, and a benchmark server to help agents
develop correct, fast GPU kernels.

<p align="center">
  <img src="docs/_static/tirx-harness-overview.png" alt="TIRx Harness architecture: the agent workflow connects to a knowledge base, compiler analyses, the TIRx foundation, and a benchmark server, with evolution traces and optimized kernels feeding self-improvement." width="85%">
</p>

TIRx Harness brings together:
* **TIRx-lite** for kernel authoring: a domain-specific language over the TIRx
  intermediate representation.
* **Compiler analysis**: check synchronization, memory races, and numerical
  behavior; inspect compiler output and generated GPU instructions.
* **[kcoral](https://kcoral.mlc.ai/)** for remote execution: keep your agent on
  one machine and run GPU work on another.

Skills guide the agent in using them, while workload contracts define
correctness and performance.

## Get Started

Install the released package from PyPI:

```bash
python -m pip install tirx-harness
```

To develop the harness, follow
[Build from source](docs/installation.md#build-from-source)
for the build prerequisites and repository checkout.

See the [documentation](docs/index.md) for details:

- [Installation](docs/installation.md): prerequisites, other installation methods, and agent skills
- [Quick Start](docs/quick-start.md): optimize a kernel in your own project
- [Optimization Runs](docs/optimization-runs.md): run a registered workload

The book [Agentic GPU Programming for MLSys](https://mlc.ai/agentic-gpu-programming-for-mlsys/)
explains the design behind the harness and walks through an optimization workflow.

## Contributing

- [Add a workload](docs/development/add-workload.md): define a new optimization task
- [Contribute a kernel](docs/development/contribute-kernel.md): publish a kernel produced by a run
- [Report and fix bugs](docs/development/report-and-fix-bugs.md): reproduce and resolve a defect

## License

TIRx Harness is licensed under the [Apache License 2.0](LICENSE).
