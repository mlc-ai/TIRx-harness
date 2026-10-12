# FA4 forward correctness validation for issue #19

The forward gate adds a normalized RMS bound of 0.2 while retaining the existing
absolute/relative tolerances and matched-ratio requirement. Historical B200
validation accepted five genuine outputs and rejected all ten zeroed or halved
outputs. The old gate accepted the two corrupted non-causal outputs.

## Scope and recorded results

[The evidence bundle](fa4_forward_issue19_b200.json) contains all 15 forward cases from the full run
and all three from the first-workload smoke run, including every recorded metric
and failure note. The original combined forward/backward metadata and summary are
retained unchanged and explicitly labeled; separately derived forward summaries
avoid presenting the combined counts as forward-only results.

| Check | Historical result |
| --- | --- |
| Full-run official workloads | 5 |
| Genuine outputs accepted | 5/5 |
| Corrupted outputs rejected by the RMS gate | 10/10 |
| Corrupted outputs accepted by the old gate | 2/10 |
| Full-run expectations met | 15/15 |
| Smoke expectations met | 3/3 |
| Smoke old-gate false positives | 0/2; the first workload is causal |
| Maximum genuine comparator normalized RMS | 0.0002953275397885591 |
| Maximum genuine independently reported per-output RMS ratio | 0.00029532753415398045 |

The two RMS values use the historical helper's different arithmetic paths and
are preserved separately. The bundle also keeps large relative-error metrics
near zero reference values, the eight corruptions already rejected by the old
gate, and all ten new-gate rejection notes. No unfavorable measurements were
removed. There were no error-kind records or failed expectations in either run.

## Hardware, software, and inputs

The test used one NVIDIA B200, SM100 / compute capability 10.0, with 183359 MiB
reported device memory and driver 595.91.07. Python was 3.12.12 and PyTorch was
2.9.1+cu128 (CUDA 12.8). Principal packages were `flash-attn-4==4.0.0b33`,
`quack-kernels==0.6.4`, and `nvidia-cutlass-dsl==4.6.2`. The bundle retains every
one of the 161 original package/version pairs. Each version carries an explicit
`version:` label; removing that label recovers the exact original string.
The driver report's CUDA 13.2 denotes driver support, not the PyTorch CUDA build.

Each q/k/v tensor has shape `[batch, sequence, 16, 128]` and dtype FP16. The
archived benchmark draws FP32 standard-normal inputs, scales them by 0.5, and
casts them to FP16. It seeds each workload using the first eight hexadecimal
digits of SHA256 over `fa4-forward:{entry_id}:{axes}`, where `entry_id` is the
first eight characters of the workload UUID and `axes` uses the original Python
dictionary representation and field order. These derived seeds are recorded in
the bundle with the original workload IDs and axes. Attention scale is
0.08838834764831843 for every workload.

| Batch | Sequence | Causal | Derived seed |
| --- | --- | --- | --- |
| 1 | 2048 | true | 2894067797 |
| 1 | 4096 | true | 378428603 |
| 2 | 4096 | true | 1071686831 |
| 1 | 8192 | true | 4007805548 |
| 1 | 8192 | false | 3917596130 |

The oracle is independent FP32 PyTorch SDPA MATH, evaluated one batch/head at a
time, with dropout disabled and TF32 disabled. The production comparator uses
ATOL=0.05, RTOL=0.05, matched ratio >=0.999, and normalized RMS limit 0.2. Its RMS
denominator adds 1e-8; the existing absolute-error floor of 1e-6 still applies.
Genuine FA4 CuTe outputs and their zeroed/halved variants are compared directly.
The validation performs correctness checks without timing or repeated-run
stability measurements.

## Reproduction

Use a B200-compatible CUDA environment with Python 3.12.12. Run from the repository
root and set `VALIDATION_ENV` and `EVIDENCE_ROOT` to writable locations outside the
checkout. The following targeted setup recreates the principal dependencies;
it is adapted from the historical setup, not a newly executed installation.
The complete historical software inventory is in the bundle.

```bash
python3.12 -m venv "${VALIDATION_ENV}"
"${VALIDATION_ENV}/bin/python" -m pip install 'torch==2.9.1+cu128' --index-url https://download.pytorch.org/whl/cu128
"${VALIDATION_ENV}/bin/python" -m pip install 'flash-attn-4==4.0.0b33' 'quack-kernels==0.6.4' 'nvidia-cutlass-dsl==4.6.2' 'numpy==2.5.3' 'pytest==8.4.2'
"${VALIDATION_ENV}/bin/python" evolution/validation/fa4_issue19_validate.py --task forward --max-workloads 1 --output "${EVIDENCE_ROOT}/forward-smoke.jsonl"
"${VALIDATION_ENV}/bin/python" evolution/validation/fa4_issue19_validate.py --task forward --output "${EVIDENCE_ROOT}/forward-full.jsonl"
"${VALIDATION_ENV}/bin/python" -m pytest evolution/tests/test_fa4_forward_rms.py evolution/tests/test_input_dependence.py -q
(cd evolution/validation && sha256sum -c fa4_forward_issue19.SHA256SUMS)
```

The [published helper](fa4_issue19_validate.py) preserves the original oracle,
comparison arithmetic, input-generation calls, variants and metric reporting.
It updates the description, requires explicit selection of one task, and records
current Git HEAD instead of reading a historical revision override. Its JSON
sequence type check uses equivalent Python 3.10+ union syntax for repository lint. The bundle
records original and published helper hashes. The historical commands used the
original helper's default combined-task mode; portable versions of those exact
recipe commands are retained separately from the new forward-only invocations.

## Provenance and limitations

GPU execution occurred on 2026-10-09 using base
`6449c3d7bf3e965c71333a2200843446310953c7` plus the task patch. The bundle's source
manifest hashes eight relevant archived source files, including the benchmark,
comparator, workload definition, input-generating workload list and CPU tests.
Each is byte-identical to PR commit `639b83dd2fb028518b92924cd82ec4112c9df46a`.
The later base update changed neither the benchmark subtree nor the tests.
This publication changes evidence and adds the reproduction helper;
**GPU validation was not rerun**, and historical results are not current CI.

Raw source and result archives, scripts and logs remain unchanged outside this
PR. Every artifact identifies its original SHA256 and describes its derivative.
The source archive hash is `8e0cca163ca341ac484e070dda56f2b9a79d561fa72c081dac4638495a3c5549`;
the result archive hash is `d7f6854211d721c225850fe5f873b28eb2c009acae6f3557e35ec3a27d731e72`.
Run records were filtered only by task; retained records and combined metadata
were not edited. Software inventory conversion is lossless. Hardware extraction
omits execution identifiers, while warning extraction replaces execution paths
with portable placeholders. The checksum file verifies the published bundle's
bytes; original hashes identify the retained raw inputs and do not claim those
raw files are publicly attached.

- The GPU run used real FA4 CuTe `_flash_attn_fwd` as a validation-only substitute.
  It did **not** run the repository's original FA2 `flash_attn_func` baseline end
  to end. That implementation is unchanged.
- GPU checks call the production comparator directly. The committed CPU tests
  separately exercise `run_suite` to `run_benchmark`; the historical combined
  regression log reports 20 passing cases across both FA4 tasks and
  input-dependence tests, not 20 forward-only cases.
- CUTLASS emitted `AuxData` / `JitArgument` warnings in both numeric runs. Their
  messages are retained, and both runs completed with all expectations met.
- Setup initially warned that NumPy was unavailable. The completed software
  inventory includes `numpy==2.5.3`; the three warning occurrences are retained.
- This targeted validation environment used CUTLASS DSL 4.6.2, whereas the root
  repository lock uses a different version. It does not validate the full locked
  repository environment. No performance or general candidate-acceptance claim
  is made.
