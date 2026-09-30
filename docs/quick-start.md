(installation-quick-start)=
# Quick Start

```{container} lead
Use the skills alongside your tests and benchmarks.
```

(before-you-start)=
(install-python-packages)=
(optional-install-locked-versions)=
(optional-install-with-uv)=
(install-agent-skills)=
## Prepare your environment

This example uses a local copy of
`tirx-kernels/tirx_kernels/gemm/fp16_bf16_gemm.py` on NVIDIA B200. It also needs
a compatible CUDA-enabled PyTorch installation.

With the [system prerequisites](installation.md#before-you-start) available,
activate the Python environment your agent uses and install the released package
from [PyPI](installation.md#install-from-pypi):

```bash
python -m pip install tirx-harness
```

Check that its packages resolve:

```bash
python -c "import tvm.tirx, tvm_ffi, tirx_kernels.tirx_lite, tirx_harness; print('Core imports OK')"
```

This checks imports only. To develop the harness itself, use
[Build from source](installation.md#build-from-source).

From your project directory, install the skills where your agent reads them:

```bash
tirx-harness skills install --dest .agents/skills
```

Use your agent's discovery convention, such as `.agents/skills` or
`.claude/skills`. The command also downloads the wiki manuals and reference
repositories.

(use-your-existing-agent)=
## Optimize a kernel

Clone `tirx-kernels` into the project directory:

```bash
git clone https://github.com/mlc-ai/tirx-kernels.git
```

Start your agent (for example, Codex or Claude Code) from the project directory.

Give your agent the following goal:

> /goal Optimize a local copy of `tirx-kernels/tirx_kernels/gemm/fp16_bf16_gemm.py` for
> FP16 `C = A @ B.T`, `M=N=K=1024`, on NVIDIA B200. Aim to match or beat
> `torch-cublas` while preserving correctness.
> Compare original, optimized, and cuBLAS timings on the same GPU with identical
> settings over three independent runs. Report latencies, speedups, and any
> remaining gap.

Read correctness before speedup. Compare original, optimized, and cuBLAS timings
on the same GPU with identical settings over three independent runs. Check the
reported latencies, speedups, and any remaining gap alongside the modified kernel.

## Run an existing workload

Follow [Optimization Runs](optimization-runs.md) for one complete example: install
the setup dependencies, select the FP16 GEMM workload, prepare a worktree,
launch an agent, and inspect the resulting frontier.
