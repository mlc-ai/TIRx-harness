# Kernel-development skills

Copy each skill as a complete directory, including its references and scripts.
Python packages are installed separately; see [Installation](../docs/installation.md)
for packages and skills, and [Quick Start](../docs/quick-start.md) for an example agent goal.

| Skill | Source | Purpose |
| --- | --- | --- |
| `tirx-wiki` | This directory | APIs, canonical kernels and reference lookup |
| `tirx-debug-kernel` | This directory | Correctness debugging and validation |
| `tirx-profile-kernel` | This directory | Performance investigation and profiling |
| `KernelWiki` | Pinned external repository in `external.json` | Additional kernel-development references |
| `ncu-report-skill` | Pinned external repository in `external.json` | NCU report analysis |

The `tirx-harness` wheel bundles the three local skills. `tirx-harness skills
install --dest DIR` copies them and runs the copied wiki's
`scripts/fetch_references.py` to download its reference materials. Run setup
selects the five explicitly through `evolution/toolsets/kda_flow.yaml`.

For optimization runs, run setup yourself before launching the agent; see
[run preparation](../evolution/README.md#run-preparation).
