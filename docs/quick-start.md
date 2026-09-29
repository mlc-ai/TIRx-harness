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
`tirx-kernels/tirx_kernels/gemm/fp16_bf16_gemm.py` on NVIDIA B200.

With the [system prerequisites](installation.md#before-you-start) available,
activate the Python environment your agent uses and install the released package
from [PyPI](installation.md#install-from-pypi):

```bash
python -m pip install tirx-harness torch \
  --extra-index-url https://download.pytorch.org/whl/cu132
```

Check that its packages resolve:

```bash
python -c "import tvm.tirx, tvm_ffi, tirx_kernels.tirx_lite, tirx_harness; print('Core imports OK')"
```

This checks imports only. To develop the harness itself, use
[Build from source](installation.md#build-from-source).

Clone the repository to get the skill files:

```bash
git clone https://github.com/mlc-ai/TIRx-harness.git
cd TIRx-harness
```

Copy the skills to the directory your agent reads in `TIRx-harness`:

```bash
skills_dir="$PWD/.agents/skills"
mkdir -p "$skills_dir"
cp -R skills/tirx-wiki skills/tirx-debug-kernel skills/tirx-profile-kernel "$skills_dir/"
(cd "$skills_dir/tirx-wiki" && python scripts/fetch_references.py)
```

Use your agent's discovery convention, such as `.agents/skills` or
`.claude/skills`. Copy each skill as a complete directory. The fetcher downloads
the wiki manuals and reference repositories; it requires network access.

(use-your-existing-agent)=
## Optimize a kernel

Clone `tirx-kernels` from the `TIRx-harness` repository root:

```bash
git clone https://github.com/mlc-ai/tirx-kernels.git
```

Start your agent (for example, Codex or Claude Code) from the
`TIRx-harness` repository root.

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
