# Workload declarations

`grouped_gemm_fp8` uses the existing Grouped GEMM
preparation and DeepGEMM baseline from `tirx-kernels` on B200.
All four configurations must pass; report per-workload and geometric-mean speedup.
The `benchmark` dependency group installs the matching DeepGEMM wheel
(Python 3.12/3.13, Linux x86_64, glibc >= 2.38). The GPU worker needs
CUDA Toolkit 13.2 for runtime JIT; set `CUDA_HOME` if it is not on PATH.
Use `uv run --package tirx-evolution evolve init --task grouped_gemm_fp8`
with `--remote URL` for KCoral; the generated prompt includes evaluation commands.

`evolution/tasks/` contains one immutable benchmark contract per task.
`evolution/toolsets/kda_flow.yaml` is the authority for the skills and shared
leak bans installed by KDA setup. The loaders live in
`evolution/preparation/declare.py`.

## Registering a task

Every workload runs through the single packaged entry point:

```text
python evolution/benchmark/adapter.py <workload_dir> vN
```

Add the task's inputs, correctness gate, timing baseline, definition, and
workload rows to `evolution/benchmark/flashinfer_bench_evolve/tasks/`, then
map its workload directory to that package task in the adapter's `PACKAGED`
table. Workload leaves contain no committed benchmark source; candidate
`v<N>/` directories are created only inside generated run worktrees.

The adapter's `PACKAGED` registry fixes each task's shape mode for both local
and remote scoring. `BENCH_OFFICIAL_SHAPE_MODE` cannot override it. Tasks
without a specific shape suffix use `all`, including any synthetic stress
rows supplied by the packaged task. Register single-shape tasks separately
with a shape suffix, as with KDA forward and backward.

Use `pinned` for selection by a reviewed UUID in
`adapter.PINNED_WORKLOAD_UUIDS`. Use `max` for the packaged benchmark's
own largest-shape selection, as the existing KDA forward/backward tasks do.
The adapter resolves `pinned` rows before passing explicit workloads to the
packaged benchmark; it delegates `all` and `max` to the package unchanged.

The six CAKE single-shape tasks use explicit dimension suffixes, following
the existing `kda_forward_b1_t8192_h96` convention:

| Workload | Single-shape task |
| --- | --- |
| Alpha-MoE | `alphamoe_m128_e512_topk10_k2048_i128_fp8` |
| KDA decode | `kda_decode_b128_t1_h16_hv32_d128_bf16` |
| MLA DSV4 prefill | `mla_dsv4_prefill_b2_qsum386_qmax257_h128_d512_swa16384_c16384_topk1152_bf16_hnd_varlen` |
| MSA prefill | `msa_prefill_b1_q4096_kv4096_hq64_hkv4_d128_topk16_bf16_flat` |
| MSA decode | `msa_decode_b128_q16_kv4096_hq64_hkv4_d128_topk16_bf16_flat` |
| VSA | `vsa_s80000_h8_d128_blk128_topk156_bf16` |

Each task's YAML filename matches its `name`. Its `workload_dir` uses the
same suffix under the existing workload family (`kda/decode_...`,
`msa/prefill_...`, and `msa/decode_...`). `hq` and `hkv` distinguish MSA
query and KV heads; KDA's `h` and `hv` distinguish Q/K and value heads.
`d`, `blk`, and `topk` denote head dimension, block size, and selection count.
For Alpha-MoE, `m`, `e`, `k`, and `i` denote tokens, experts, hidden size,
and intermediate size; `fp8` refers to weights and quantized computation,
with bf16 activation inputs.

Alpha-MoE, KDA decode, MSA prefill/decode and VSA use fixed sequence lengths.
For fixed attention tasks, both Q and KV lengths are uniform across
requests; a uniform `cu_seqlens` array is compatible with this rule.

MLA DSV4 explicitly selects the large varlen prefill row. Its `qsum386` and
`qmax257` suffixes mean 386 total queries and at most 257 per request; the
actual Q lengths are [129,257]. The SWA and compressed KV pools have base
lengths 16384, with actual lengths [16384,16640] and [16384,16448]. All KV
lengths fill complete pages. `topk1152` is the sparse width including 128
SWA slots; active sparse lengths vary from 832 to 1152. Query and pools
use bf16/HND. The row has 56,918,016 nominal query/key pairs across heads
and 49,545,216 valid pairs. Its 4K-pool counterpart has the same pair counts;
the selected 16K row uses larger KV pools. This task retains the original
varlen contract and published seed.

`nvfp4_attention` already contains one row and has no duplicate. Each
single-shape declaration states the selected official row and its full
contract. The six choices favor substantial work within the intended
contract and baseline arm, with consistent storage. Power-of-two block
counts are not required.
All six tasks use the `pinned` mode. `adapter.PINNED_WORKLOAD_UUIDS`
selects their reviewed official rows for both local and remote scoring.
Pinned selection excludes synthetic stress rows and requires exactly one
matching official UUID. It preserves original row metadata, including seeds,
and uses the full official list regardless of `BENCH_INCLUDE_OFFICIAL` or
`BENCH_MAX_OFFICIAL`. Other tasks keep their existing shape selection.

Declare the optimization contract in `evolution/tasks/<name>.yaml`:

```yaml
name: my_kernel
workload_dir: candidates/my_kernel
kernel_authoring: TIRx-lite
sota_baseline:
  name: baseline_name
bench_timeout_s: 120
banned_paths:
  - .claude/skills/tirx-wiki/references/repos/baseline-source/**
spec: |
  Complete immutable math, tensor, interface, correctness, and scoring
  contract for the workload.
```

`name`, `workload_dir`, `kernel_authoring`, `sota_baseline.name`, and a
non-empty `spec` are required. `kernel_authoring` is `TIRx-lite` for the shared TIRx-lite
contract or `task` when the task spec defines another implementation language.
`bench_timeout_s`, when present, must be positive.

## Leak bans

- Ban the workload's own SOTA source and any prior solution corpus.
- Setup also bans the source checkout and other run directories, while
  allowing the current run to access its own files.
- Source carrying a reference implementation is physically removed from the
  generated worktree; the hook is an additional access boundary, not a
  substitute for removal.
- Shared prior-solution corpora belong in `evolution/toolsets/kda_flow.yaml`;
  workload-specific bans belong in the task YAML.

## Model weights

`HF_MODEL_PATH` is the harness's model root on the machine executing the benchmark;
it defaults to `/raid/catalyst/models`. Set it in the local benchmark shell
or, for remote runs, the GPU worker's startup environment. Remote clients
submit the task and candidate without needing the worker's filesystem paths.

Each workload uses its declared subdirectory under this root. Existing files
are read directly; missing files are downloaded from Hugging Face at the
workload's pinned revision into the same subdirectory. This also completes
partial downloads on a later invocation. Reading a complete model needs no
network access or filesystem writes. `HF_HOME` and `HF_HUB_CACHE` do not select
the benchmark's model directory.

Each model's section below declares its subdirectory, source, revision and
required files. Provisioned files must match that revision; existing files
are reused without version or integrity checks. Install the `benchmark`
dependency group on the GPU worker.

Downloads require network access and a writable model directory. To retain
weights across requests, keep the model root on persistent storage visible
to each worker. A read-only, offline worker must have the complete model
mounted before running. Downloads count toward the request timeout, outside
the GPU score. For KCoral's server configuration example, see
[Model weights on the worker](../../docs/components/kcoral.md#model-weights-on-the-worker).

## Qwen3.8 full-model workloads

These six TIRx-lite tasks score a prepared Qwen3.8-27B forward, including
all 64 layers, full-vocabulary logits and cache/state writes:

| Workload | Multi-shape (13 cases) | Single-shape | Selected case (batch, previous, new) |
| --- | --- | --- | --- |
| Decode | `qwen38_decode` | `qwen38_decode_b128_p4096_n1` | D4: (128, 4096, 1) |
| Prefill | `qwen38_prefill` | `qwen38_prefill_b1_p0_n32768` | P3: (1, 0, 32768) |
| Expand | `qwen38_expand` | `qwen38_expand_b64_p8192_n256` | S6: (64, 8192, 256) |

`p` counts existing prefix tokens and `n` counts new tokens per request.
The selected rows cover ordinary decode, long fresh prefill and cached
expansion. Multi-shape decode also includes four-position verification;
multi-shape expand includes the three heterogeneous request batches.
The single-shape tasks use the adapter's existing `pinned` selection.

The shared `qwen38/shapes.py` and independent `qwen38/model.py` are imported
from [qwen38-inference at 05e8ef0](https://github.com/mlc-ai/qwen38-inference/tree/05e8ef0db302a2c005994b79d9a5e78b96e2d88b).
The model imports Gemma RMSNorm directly from FlashInfer, the same BF16
implementation used by the upstream SGLang wrappers, avoiding their compiled
Torch ABI dependency. Its input/cache helpers are adapted in `qwen38/baseline.py`; the three task
modules share `qwen38/benchmark.py`. Upstream licenses accompany the model.
Optimized upstream solutions are not part of the benchmark package.

These tasks follow the [model weights convention](#model-weights):

| Model setting | Value |
| --- | --- |
| Subdirectory | `Qwen3.8-27B` |
| Hugging Face repository | `Qwen/Qwen3.8-27B` |
| Pinned revision | `MODEL_REVISION` in [qwen38/model.py](../benchmark/flashinfer_bench_evolve/tasks/qwen38/model.py) |

The model directory must contain `config.json`, `generation_config.json`,
`model.safetensors.index.json` and the indexed shards. A B200, CUDA 13 C++
toolchain and sufficient GPU/host memory for the 27B weights, KV cache and
CPU state snapshots are required.

Each task invocation loads the model once and reuses its weights across all
selected shapes. Each shape prepares its own real prefix and cache. Separate
task invocations load separate model instances. Set `QWEN38_SEED` to reproduce
the same inputs across invocations; by default each invocation draws a new seed.
CPU snapshots are copied directly into pinned memory; initial all-zero states
are restored by zeroing instead of saving/transferring a snapshot.

```bash
uv run --package tirx-evolution evolve init --task qwen38_decode_b128_p4096_n1
# In the prepared run, use the generated prompt's benchmark command.
python evolution/benchmark/adapter.py candidates/qwen38/decode_b128_p4096_n1 baseline
```

Candidates export `setup(data) -> callable`. The YAML states the task contract
and points to `benchmark.tirx_prepare`, `model.Qwen38._prepare_step` and
`model.Qwen38._forward` for model data, input/cache layouts and computation.
Each call computes current logits and state with TIRx-lite. Prefix preparation,
compilation, capture and state restoration are untimed. Decode uses outer CUDA
graph replay; prefill/expand
time the prepared callable directly. Scoring uses GPU-event samples
and the upstream per-element `atol=rtol=0.001` gate on logits and caches.
The harness checks full logits and all cache tensors once before timing, then
releases those reference snapshots. Each implementation starts its timing
block from the same saved state and runs 3 warmup calls followed by 9 retained
samples without intermediate restoration. Prepared tokens, positions and
lengths stay fixed while cache state advances; this measures repeated prepared
calls, not continuous generation. Final block logits must remain finite, but
are not compared to the single-step reference. `BENCH_TRIALS` defaults to 1;
additional trials reverse implementation order and restore before every block.
Baseline self-checks use the reference in both blocks. Prefix preparation,
snapshots, transfers and correctness checks remain outside the GPU score.
