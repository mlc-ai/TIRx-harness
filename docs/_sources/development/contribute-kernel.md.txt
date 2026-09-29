# Contribute a kernel

```{container} lead
Package a selected run result, validate it, and submit it to tirx-kernels.
```

Kernel contributions go to the {kernels}`tirx-kernels repository <README.md>`.
Start with a passing candidate from the run's frontier; see
[Inspect the results](../optimization-runs.md#inspect-the-results) to select one.

## Package the candidate

Work in a development checkout of `tirx-kernels`. Adapt the selected candidate
into `tirx_kernels/<workload>/<name>.py`, removing dependencies on the run
directory and other candidates. Document its supported workload, source run,
candidate name, and revision in the module.

Use the kernel's descriptive name, such as `kda_decode_multishape`, as its
globally unique `KERNEL_META["name"]`. Ports from external projects belong under
`tirx_kernels/ported/<upstream>/`, with their upstream attribution and license.

Prepare these package changes, following a nearby kernel:

| Change | Purpose |
| --- | --- |
| Kernel module | Provide the metadata, configurations, and hooks defined by the {kernels}`kernel protocol <tirx_kernels/_protocol.py>`. The registry discovers kernels through their `KERNEL_META` declarations. |
| Benchmark configuration | Add `config/<workload>/<name>.yaml` under the {kernels}`benchmark suite <tirx_kernels/bench_suite>`, with `kernel` matching the public registry name. Ports use `config/ported/<upstream>/`. |
| Catalog entry | Link the kernel from the {kernels}`kernel catalog <README.md#native-tirx>`. |

## Validate and record results

Validate the packaged implementation through its correctness runner across
the configurations and architectures it claims to support. This checks the
code users will import after integration. Use the applicable
[analysis tools](../components/tools.md) when changing synchronization, memory
access, or numerical behavior.

Measure the packaged kernel and its reference in the same run, with the same
GPU, inputs, dependencies, and timing method. For changes to an existing
kernel, also compare before and after using the
{kernels}`benchmark suite <tirx_kernels/bench_suite/README.md>`.

Save the reproduction commands, environment,
and raw correctness and timing reports as the evidence for this contribution.
Follow the benchmark suite's promotion procedure when updating its baseline.

## Open a pull request

Submit the package changes to `tirx-kernels`.
Describe the implementation idea and its limitations, and link the validation
evidence from the previous step. Keep run logs and intermediate candidates
outside the kernel package.

For a harness or tool defect found during the run, follow
[Report and fix bugs](report-and-fix-bugs.md).
