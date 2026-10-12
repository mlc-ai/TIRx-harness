# FA4 backward correctness evidence for issue 19

The backward gate now applies a normalized RMS limit of `0.2` independently to
`dq`, `dk`, and `dv`. It retains the absolute tolerance `0.1`, relative tolerance
`0.1`, and required matched ratio `0.995`. This prevents small gradients from
being accepted merely because zero or half-amplitude outputs meet the absolute
tolerance. The existing comparator rejects RMS ratios at or above the limit
when that output has maximum absolute error greater than `1e-6`; its RMS
denominator includes the original `1e-8` stabilizer.

## Historical GPU results

The preserved run used one NVIDIA B200, SM100 (compute capability 10.0),
183359 MiB of device memory, driver 595.91.07, Python 3.12.12, and
PyTorch 2.9.1+cu128. It used `flash-attn-4==4.0.0b33`,
`quack-kernels==0.6.4`, and `nvidia-cutlass-dsl==4.6.2`.
The bundle contains all 161 recorded package versions.

| Backward numeric check | Smoke | Full |
| --- | --- | --- |
| Official workloads | 1 | 5 |
| Genuine output triples accepted | 1/1 | 5/5 |
| Zeroed/halved triples rejected by the new gate | 8/8 | 40/40 |
| Corrupted triples accepted by the old gate | 8/8 | 40/40 |
| Expectations met | 9/9 | 45/45 |

The maximum genuine normalized RMS reported by the production comparator was
`0.00030322762904688716`. The separately calculated per-output maximum was
`0.0003032276210305715`; both original fields are retained without rounding in
[the evidence bundle](fa4_backward_issue19_b200.json).

The GPU checks called the unmodified `fa4_backward.baseline.run`
(`_flash_attn_bwd`) with the task's original FP16 input generation. The independent
reference used FP32 PyTorch SDPA MATH, per batch/head to bound memory, with
autograd for gradients. TF32 was disabled. Each workload tested genuine outputs,
zeroing each gradient and all three together, and halving each gradient and all
three together. All old/new gate metrics, per-output gradient metrics, failure
notes, and expected verdicts are preserved for all 45 full and 9 smoke cases.

Inputs have shape `(batch, sequence, heads, dimension)`, FP16 dtype, Q/K/V scale
`0.5`, and output-gradient scale `0.25`. The bundle includes every workload's
shape, causal flag, and derived seed. The archived generator computes the seed
from the first eight hexadecimal SHA256 digits of
`f"fa4-backward:{entry['id']}:{entry['axes']}"`, where `id` is the workload UUID's
first eight characters. This preserves the original dictionary ordering.

## Reproduce the checks

Use a separate Python 3.12 environment with a working CUDA-enabled
`torch==2.9.1+cu128` installation on B200. These commands use the published
helper and the recorded targeted dependencies; they do not replace the project's
full dependency manifest or claim to reproduce its complete locked environment.
Run from the repository root:

```bash
python -m pip install 'flash-attn-4==4.0.0b33' 'quack-kernels==0.6.4' \
  'nvidia-cutlass-dsl==4.6.2' 'numpy==2.5.3' 'pytest==8.4.2'
EVIDENCE_ROOT="$(mktemp -d)"
python evolution/validation/fa4_issue19_validate.py --task backward --preflight
python evolution/validation/fa4_issue19_validate.py --task backward \
  --max-workloads 1 --output "${EVIDENCE_ROOT}/backward-smoke.jsonl"
python evolution/validation/fa4_issue19_validate.py --task backward \
  --output "${EVIDENCE_ROOT}/backward-full.jsonl"
python -m pytest evolution/tests/test_fa4_backward_rms.py \
  evolution/tests/test_input_dependence.py -q
```

The helper calls the production comparator. The committed CPU regressions
separately exercise `run_suite` through `run_benchmark`, including which gradient
is rejected. The historical combined CPU run passed 20 tests: four forward,
ten backward, and six input-dependence cases. Its unchanged log is bundled;
these counts are not presented as a new run of this branch.

## Provenance and integrity

The measured source was revision
`6449c3d7bf3e965c71333a2200843446310953c7` plus the archived task patches.
The bundle records six source SHA256 values from the source archive, covering
the adapter, shared comparator, backward benchmark/baseline, definition, and
workloads. Each matches the corresponding implementation in this PR. The
benchmark hash is
`8bc02825df3fb4e685a529559eb373943a6e530c582d05388ea1518e41742fc8`.

The original source archive SHA256 is
`8e0cca163ca341ac484e070dda56f2b9a79d561fa72c081dac4638495a3c5549`;
the result archive SHA256 is
`d7f6854211d721c225850fe5f873b28eb2c009acae6f3557e35ec3a27d731e72`.
Original archives, scripts, JSONL, and logs remain unchanged outside the PR.
The bundle identifies each retained artifact's original byte hash and any
representation changes. It also records the original validator hash and the
published helper hash.

The JSONL artifacts are task-filtered derivatives of combined forward/backward
runs. Every backward case is structurally unchanged. Original combined metadata
and summaries are labeled and retained unchanged, so combined counts of 60 full
and 12 smoke cases must not be mistaken for backward-only counts. Derived
backward summaries are separate. Diagnostic excerpts retain all numerical-run
warnings with portable installation prefixes. The complete software inventory
uses an explicit `version: ` prefix in each version string to distinguish
software versions from network addresses; removing that prefix reconstructs
every original package/version pair exactly. Hardware metadata excludes device
addresses, identifiers, and execution paths.

Verify the submitted bundle from this directory:

```bash
sha256sum -c fa4_backward_issue19.SHA256SUMS
```

## Limitations

GPU validation has not been rerun for this evidence packaging or the published
helper adaptations. The helper now requires an explicit task, records the
current checkout revision rather than an optional historical override, updates
its docstring, and uses equivalent Python 3.10+ union syntax in JSON type checks.
Its oracle arithmetic, corruption construction, and comparator calls are
unchanged. These historical measurements are not current CI results.

The historical combined runs emitted CUTLASS warnings that `AuxData` could not
be converted to `JitArgument` for forward and backward kernels. Both completed
with all expectations met; the warnings are preserved. Setup initially emitted
missing-NumPy warnings; the final inventory records NumPy 2.5.3. The targeted
CUTLASS DSL version was 4.6.2, while the repository lock selected 4.7.0, so this
run does not validate that complete locked environment. No performance was
measured, and no performance claim is made. The forward implementation's
separate baseline substitution is recorded in the unchanged combined metadata;
it does not change this run's use of the unmodified backward baseline.
