# Installation

```{container} lead
Install the components for your own agent workflow or a kcoral GPU server.
```

For [optimization runs](optimization-runs.md), setup prepares the packages and
skills automatically.

## Before you start

- Linux x86_64, Python 3.12 or 3.13, and pip 25.1 or later.
- The CUDA Toolkit (`nvcc`, `ptxas`) where GPU kernels are compiled, and a
  compatible NVIDIA driver where they are run.
- Cargo, **Rust 1.89.0 or later**, and a C linker for numerical simulation and
  correctness checks with either installation method.

## Install Python packages

Activate your agent's Python environment and choose one method.

### Install from PyPI

Install the released package without cloning this repository:

```bash
python -m pip install tirx-harness
```

Wheels include the native frontend. If pip builds from a source distribution,
the build tools below are required.

### Build from source

Source builds require Git, C/C++ build tools, Python development headers, and
the Rust toolchain listed above.

```bash
git clone https://github.com/mlc-ai/TIRx-harness.git
cd TIRx-harness
git submodule update --init thirdparty/tvm-rust-ext
python -m pip install .
```

#### Optional: install with uv

From the initialized checkout, build the harness with dependencies from `uv.lock`:

```bash
uv sync --locked
source .venv/bin/activate
```

### Verify the installation

After any of these methods, check imports (this does not run checks or GPU kernels):

```bash
python -c "import tvm.tirx, tvm_ffi, tirx_kernels.tirx_lite, tirx_harness; print('Core imports OK')"
```

## Install kcoral server dependencies

From the repository root, run `python -m pip install --group server`, or
`uv sync --locked --only-group server` and activate `.venv`. These install only
the server dependencies. See {ref}`remote execution <select-remote-execution>`
to start the server.

## Install agent skills

| Skill | Purpose |
| --- | --- |
| {repo}`tirx-wiki <skills/tirx-wiki/SKILL.md>` | Find TIRx-lite APIs, canonical kernels, and GPU references. |
| {repo}`tirx-debug-kernel <skills/tirx-debug-kernel/SKILL.md>` | Check correctness, investigate findings, and verify fixes. |
| {repo}`tirx-profile-kernel <skills/tirx-profile-kernel/SKILL.md>` | Measure performance and use profiler evidence to guide changes. |

Clone this repository if needed, then copy the skills to your agent's directory
(such as `.agents/skills` or `.claude/skills`). Run from the repository root:

```bash
skills_dir=/absolute/path/to/your/project/.agents/skills
mkdir -p "$skills_dir"
cp -R skills/tirx-wiki skills/tirx-debug-kernel skills/tirx-profile-kernel "$skills_dir/"
(cd "$skills_dir/tirx-wiki" && python scripts/fetch_references.py)
```

The fetcher needs network access to download the wiki manuals and reference
repositories, including `tirx-kernels`.

Continue to [Quick Start](quick-start.md) for a concrete example.
