# fp16 GEMM correctness gate: issue #22

The 99% matched-element gate accepted both race kernels supplied in
[issue #22](https://github.com/mlc-ai/TIRx-harness/issues/22), despite sparse large
errors. This change requires 99.99% matching, retains `ATOL=0.1` and `RTOL=0.01`,
and updates the task description. The shared comparator marks an element bad
when **both** its absolute and relative errors exceed their thresholds; its
comparison formula is unchanged. The task ratio applies to preflight, observed
timing outputs, post-timing checks, and input-dependence checks.

Ten CPU regression cases cover sparse corruptions that passed the old gate,
allowed-outlier boundaries, exact and tolerated outputs, and the actual task's
`run_suite` integration. The relevant CPU selection has 16 passing cases:

```bash
python -m pytest -q evolution/tests/test_fp16_gemm_floor_task.py \
  evolution/tests/test_input_dependence.py
```

## B200 measurements

The retained [B200 bundle](fp16_gemm_floor_issue22_b200.json) contains all 11 scores,
132 official workload rows, stdout traces, exact candidate sources, runner/setup
scripts, and software records. [SHA256SUMS](SHA256SUMS) pins the bundle bytes.
Every score covered all 12 official shapes, with no `ERROR` or unrelated failure.

| Candidate | Matched ratio required | Passed workloads | Worst matched ratio | Maximum absolute error |
| --- | ---: | ---: | ---: | ---: |
| baseline | 0.9999 | 12/12 | 1.0 | 0.0 |
| race_a, legacy | 0.99 | 12/12 | 0.99920654296875 | 40.0 |
| race_b, legacy | 0.99 | 12/12 | 0.9985542297363281 | 30.8671875 |
| race_a, round 1 | 0.9999 | 6/12 | 0.9995627403259277 | 28.453125 |
| race_b, round 1 | 0.9999 | 6/12 | 0.999354362487793 | 24.4453125 |
| race_a, round 2 | 0.9999 | 6/12 | 0.9996716380119324 | 32.3125 |
| race_b, round 2 | 0.9999 | 6/12 | 0.9993009567260742 | 25.169921875 |
| race_a, round 3 | 0.9999 | 7/12 | 0.9993756413459778 | 30.703125 |
| race_b, round 3 | 0.9999 | 6/12 | 0.9993782043457031 | 22.2734375 |
| sync_a | 0.9999 | 12/12 | 1.0 | 0.0 |
| sync_b | 0.9999 | 12/12 | 1.0 | 0.5 |

Both race kernels were rejected overall in all three new-gate scores; every
failure was a matched-ratio failure. Input-dependence rechecks also caught race_a
in rounds 2 and 3. The sync controls restore only the missing loop-end `sync()` in
the [original reproduction](https://github.com/mlc-ai/TIRx-harness/issues/22#issuecomment-6103059338).
Their 1.0 matched ratios mean all elements met the original tolerances; sync_b
is not bitwise identical to the reference. Only these two supplied kernels were
evaluated. The new gate still permits up to 0.01% out-of-tolerance elements and
cannot guarantee detection of every race.

## Provenance and protocol

These are recorded GPU measurements from 2026-10-11 UTC, not a new CI run. The
GPU checkout used base revision `6449c3d7bf3e965c71333a2200843446310953c7`
with the two production edits described above. The submitted benchmark, task
contract, baseline, workload file, shared comparator, and adapter match all six
per-score source fingerprints. The remote source archive excluded `.git`;
revision and file hashes establish provenance. GPU tests were not rerun while
isolating the worktree or preparing this evidence bundle.

Device: NVIDIA B200, compute capability 10.0, compiled target `sm_100a`, driver
580.167.08. Python 3.12.3; Torch 2.9.1+cu130; toolkit `ptxas` 13.0.88;
apache-tvm 0.27.0.post1; apache-tvm-ffi 0.1.14.post0; tirx-kernels 0.1.2.post1;
nvidia-cutlass-dsl 4.7.0; NVRTC 13.2.78. The container image was
`runpod/pytorch:1.0.7-cu1300-torch291-ubuntu2404-cluster`.

Inputs and outputs use FP16. `torch.manual_seed(0)` creates A with shape `(M,K)`
and B with shape `(N,K)`; the reference computes `A @ B.T` with cuBLAS. Official
`(M,N,K)` shapes are:

```text
(1024,1024,1024)  (2048,2048,2048)  (4096,4096,4096)  (8192,8192,8192)
(2048,11008,4096) (2048,4096,11008) (4096,11008,4096) (4096,4096,11008)
(1024,8192,2048)  (8192,1024,2048)  (4096,2048,8192)  (2048,8192,1024)
```

Each score uses 10 warmup calls, 50 timing iterations, and 3 trials; 5 correctness
runs before timing and 5 after timing, an observed timing-output check, and an
input-dependence check. Successful rows record 36 checks. Input perturbation uses
`trial_idx + tensor_index` as its seed. Repeatability checking is disabled; the
three race scores are independent invocations. Rows preserve absolute, relative,
RMS and matched-ratio metrics, verdicts, timing method, latencies and speedups.
Gradients are not applicable to this GEMM task.

FlashInfer was unavailable, so timing fell back to CUDA events. No performance
claim is made from these timings. Pip reported that Torch requires NVRTC 13.0.48
while 13.2.78 was installed for the compiler dependency. The bundle retains this
warning. Imports, a real cuBLAS preflight, baseline and both sync controls passed,
and all scores completed without runtime errors. This is a working environment
for the controlled numerical comparison, not a dependency-conflict-free install
or the complete benchmark dependency group.

A prior broader local CPU selection had 35 passes and one existing grouped-GEMM
setup failure because `tirx_kernels` was absent locally. That broader selection
was not rerun for submission; the 16 relevant CPU tests were rerun in the #22
worktree. An earlier old-threshold mutation check had 5 failures and 5 passes,
confirming that the new regression tests detect the loose gate.

## Reproducing the measurements

Use a fresh checkout at the tested base revision on a compatible B200 environment.
Keep an absolute copy of the submitted bundle before checking out that revision.
The following extracts the retained scripts and exact candidate sources; no
installation or cloud rental is performed by extraction:

```bash
export ISSUE22_BUNDLE=/absolute/path/fp16_gemm_floor_issue22_b200.json
git checkout 6449c3d7bf3e965c71333a2200843446310953c7
python3 - <<'PY'
import json
import os
from pathlib import Path
bundle = json.loads(Path(os.environ['ISSUE22_BUNDLE']).read_text())
for name, artifact in bundle['artifacts'].items():
    if not name.startswith('repro/'):
        continue
    target = Path('.local/issue22') / name.removeprefix('repro/')
    target.parent.mkdir(parents=True, exist_ok=True)
    content = artifact['content']
    target.write_text(json.dumps(content, indent=2) + '\n'
                      if artifact['format'] == 'json' else content)
p = Path('evolution/benchmark/flashinfer_bench_evolve/tasks/fp16_gemm_floor/benchmark.py')
p.write_text(p.read_text().replace('REQUIRED_MATCHED_RATIO = 0.99\n',
                                  'REQUIRED_MATCHED_RATIO = 0.9999\n'))
p = Path('evolution/tasks/fp16_gemm_floor.yaml')
p.write_text(p.read_text().replace('at least 99% of', 'at least 99.99% of'))
PY
python3 -m venv --system-site-packages .local/issue22/venv
ISSUE22_PYTHON="$PWD/.local/issue22/venv/bin/python" bash .local/issue22/setup_b200.sh
source .local/issue22/b200_env.sh
"$ISSUE22_PYTHON" .local/issue22/run_initial.py --output .local/issue22/b200_initial
for round in 2 3; do
  for candidate in race_a race_b; do
    "$ISSUE22_PYTHON" .local/issue22/gpu_validate.py \
      --candidate "$candidate" --threshold task \
      --output ".local/issue22/b200_repeat${round}/${candidate}_task"
  done
done
```

Use fresh output directories. The initial batch performs the seven baseline,
legacy, race and restored-sync scores. The runner records the actual task constant,
protocol, returned rows and source hashes. Its exit status checks the expected
numerical outcome; the upstream adapter CLI exit status alone is insufficient.
Each text artifact is retained verbatim. JSON artifacts retain all decoded keys
and measured values, with `original_sha256` identifying their original file bytes.
