"""Shared input/oracle cases for canonical kernels, including failing gates.

This module owns exact verdicts and finding counts.  Source rationale for its
non-clean entries lives in ``canonical_verdict_rationale.md``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from tests.numsim.corpus.kernels.activation import (
    prepare_act_and_mul_case,
    prepare_silu_nvfp4_experts_case,
)
from tests.numsim.corpus.kernels.attention import (
    prepare_flash_attention4_case,
    prepare_flash_attention4_fp4_case,
    prepare_flash_attention_backward_case,
)
from tests.numsim.corpus.kernels.cudnn import (
    prepare_cudnn_bsa_backward_blk128_case,
    prepare_cudnn_bsa_backward_blk64_case,
    prepare_cudnn_bsa_forward_combine_blk64_case,
    prepare_cudnn_bsa_forward_blk128_case,
    prepare_cudnn_bsa_forward_blk64_case,
    prepare_cudnn_csa_compressor_fwd_case,
    prepare_cudnn_dense_blockscaled_dsrelu_quant_case,
    prepare_cudnn_dense_blockscaled_srelu_quant_case,
    prepare_cudnn_dense_blockscaled_swiglu_quant_case,
    prepare_cudnn_dense_gemm_persistent_swiglu_case,
    prepare_cudnn_dsa_sparse_attention_backward_case,
    prepare_cudnn_gdn_bprop_case,
    prepare_cudnn_gdn_prefill_case,
    prepare_cudnn_gdn_recompute_case,
    prepare_cudnn_gdn2_bprop_case,
    prepare_cudnn_gdn2_prefill_case,
    prepare_cudnn_gdn2_recompute_case,
    prepare_cudnn_gemm_proj_rope_mxfp8_bf16in_case,
    prepare_cudnn_gemm_proj_rope_mxfp8_mxfp8in_case,
    prepare_cudnn_kda_bprop_case,
    prepare_cudnn_moe_bf16_dglu_dbias_case,
    prepare_cudnn_moe_dglu_dbias_case,
)
from tests.numsim.corpus.kernels.deepgemm import (
    prepare_fp4_mqa_case,
    prepare_fp8_mqa_case,
    prepare_mega_moe_case,
    prepare_paged_mqa_case,
    prepare_tf32_hc_case,
)
from tests.numsim.corpus.kernels.flashmla import (
    prepare_sparse_decode_head64_case,
    prepare_sparse_prefill_case,
)
from tests.numsim.corpus.kernels.flex_attention import (
    prepare_flex_attention_backward_case,
    prepare_flex_attention_forward_hd256_case,
    prepare_flex_attention_forward_sm103_case,
)
from tests.numsim.corpus.kernels.gemm import (
    prepare_bmm_fp8_rubin_case,
    prepare_cudnn_dense_blockscaled_amax_case,
    prepare_fp16_bf16_case,
    prepare_fp8_bmm_case,
    prepare_fp8_blockwise_case,
    prepare_grouped_fp8_case,
    prepare_k_grouped_fp8_case,
    prepare_m_grouped_masked_fp8_case,
    prepare_nvfp4_case,
)
from tests.numsim.corpus.kernels.blockscaled_gemm import (
    prepare_fastcu_nvfp4_gemm_case,
    prepare_dense_blockscaled_rubin_case,
    prepare_grouped_masked_rubin_case,
    prepare_gather_swiglu_rubin_case,
)
from tests.numsim.corpus.kernels.normalization import prepare_rmsnorm_case
from tests.numsim.corpus.kernels.normalization import (
    prepare_flashinfer_add_rmsnorm_fp4quant_case,
    prepare_flashinfer_fused_add_rmsnorm_case,
    prepare_flashinfer_fused_add_rmsnorm_quant_case,
    prepare_flashinfer_fused_dit_layernorm_case,
    prepare_flashinfer_layernorm_case,
    prepare_flashinfer_qk_rmsnorm_case,
    prepare_flashinfer_rmsnorm_fp4quant_case,
    prepare_flashinfer_rmsnorm_case,
    prepare_flashinfer_rmsnorm_quant_case,
)
from tests.numsim.corpus.kernels.msa import (
    prepare_msa_sparse_atten_fwd_combine_case,
    prepare_msa_sparse_atten_fwd_case,
    prepare_msa_sparse_atten_fwd_nvfp4_kv_case,
    prepare_msa_sparse_prepare_flat_schedule_case,
    prepare_msa_sparse_prepare_fwd_split_atomic_case,
)
from tests.numsim.corpus.kernels.topk import (
    prepare_fast_topk_clusters_case,
    prepare_filtered_topk_case,
    prepare_radix_topk_multi_cta_case,
    prepare_radix_topk_single_cta_case,
    prepare_stable_sort_topk_by_value_case,
)
from tests.numsim.corpus.kernels.quantization import (
    prepare_mxfp4_quantize_case,
    prepare_mxfp8_quantize_case,
    prepare_nvfp4_quantize_case,
    prepare_nvfp4_quantize_per_token_case,
)
from tests.numsim.corpus.kernels.recurrent import (
    prepare_native_kda_forward_case,
    prepare_gdn_cp_prefill_case,
    prepare_gdn_decode_ilp4_case,
    prepare_gdn_decode_fp32_mtp_warp_case,
    prepare_gdn_decode_mtp_case,
    prepare_gdn_decode_t1_case,
    prepare_gdn_prefill_case,
    prepare_recurrent_kda_grouped_case,
    prepare_recurrent_kda_one_warp_case,
)
from tests.numsim.corpus.kernels.state_update import (
    prepare_selective_state_update_mtp_horizontal_case,
    prepare_selective_state_update_mtp_simple_case,
    prepare_selective_state_update_mtp_vertical_case,
    prepare_selective_state_update_stp_horizontal_case,
    prepare_selective_state_update_stp_simple_case,
    prepare_selective_state_update_stp_vertical_case,
)
from tests.numsim.corpus.kernels.native_moe import prepare_native_alphamoe_case
from tests.numsim.corpus.kernels.native_multishape import (
    prepare_native_msa_case, prepare_native_vsa_case, prepare_native_kda_decode_case,
    prepare_native_msa_decode_case, prepare_native_mla_case,
)
from tests.numsim.corpus.kernels.native_kda_backward import prepare_native_kda_backward_case
from tests.numsim.corpus.kernels.cascade import prepare_merge_state_case
from tirx_harness.numsim.cases import NumSimCase


@dataclass(frozen=True)
class CanonicalKernelCase:
    name: str
    prepare: Callable[[], NumSimCase]
    expected_verdict: str = "clean"
    numsim_review_count: int | None = None
    synccheck_phase_expected_verdicts: tuple[str, ...] = ()
    racecheck_expected_verdict: str | None = None
    racecheck_phase_expected_verdicts: tuple[str, ...] = ()
    racecheck_review_kinds: frozenset[str] = frozenset()
    racecheck_review_counts: tuple[tuple[str, int], ...] = ()
    racecheck_phase_review_counts: tuple[tuple[tuple[str, int], ...], ...] = ()
    engine_max_workers: int = 16
    synccheck_max_diagnostic_bytes: int = 16 * 1024 * 1024


def _flash_attention4() -> NumSimCase:
    return prepare_flash_attention4_case(
        batch_size=1,
        seq_len=256,
        num_qo_heads=32,
        num_kv_heads=32,
        head_dim=128,
        is_causal=False,
        seed=3000,
    )


def _flash_attention_backward_sm100() -> NumSimCase:
    return prepare_flash_attention_backward_case(is_causal=False, seed=20260806)


def _dense_fp4_mqa() -> NumSimCase:
    return prepare_fp4_mqa_case(
        seq_len=32,
        seq_len_kv=512,
        num_heads=64,
        head_dim=128,
        logits_dtype="float32",
        compressed_logits=False,
        disable_cp=True,
        seed=4100,
        num_sms=2,
    )


def _dense_fp8_mqa() -> NumSimCase:
    return prepare_fp8_mqa_case(
        seq_len=32,
        seq_len_kv=256,
        num_heads=64,
        head_dim=128,
        logits_dtype="float32",
        compressed_logits=False,
        disable_cp=True,
        seed=4200,
        num_sms=2,
    )


def _tf32_hc() -> NumSimCase:
    return prepare_tf32_hc_case(m=13, n=24, k=128, num_splits=1, seed=3200)


def _flash_mla_sparse_fwd() -> NumSimCase:
    return prepare_sparse_prefill_case(
        "flash_mla_sparse_fwd", s_q=1, s_kv=64, topk=64, d_qk=512, h_q=64
    )


def _sparse_prefill_head64() -> NumSimCase:
    return prepare_sparse_prefill_case(
        "sparse_flashmla_prefill_head64_phase1",
        s_q=1,
        s_kv=128,
        topk=128,
        d_qk=512,
        h_q=64,
    )


def _sparse_prefill_head128() -> NumSimCase:
    return prepare_sparse_prefill_case(
        "sparse_flashmla_prefill_head128_phase1",
        s_q=1,
        s_kv=128,
        topk=128,
        d_qk=576,
        h_q=128,
    )


def _sparse_prefill_head128_small_topk() -> NumSimCase:
    return prepare_sparse_prefill_case(
        "sparse_flashmla_prefill_head128_small_topk_phase1",
        s_q=2,
        s_kv=128,
        topk=128,
        d_qk=512,
        h_q=128,
    )


def _fp16_gemm() -> NumSimCase:
    return prepare_fp16_bf16_case("fp16", 256, 2048, 64, seed=2000)


def _fp8_blockwise() -> NumSimCase:
    return prepare_fp8_blockwise_case(16, 256, 512, True, seed=2800)


def _grouped_fp8() -> NumSimCase:
    return prepare_grouped_fp8_case(4, 256, 384, 512, seed=1)


def _nvfp4() -> NumSimCase:
    return prepare_nvfp4_case(256, 256, 256, seed=2900)


CANONICAL_KERNEL_CASES = (
    CanonicalKernelCase(
        "alphamoe_fp8_blockscale_qwen3next", prepare_native_alphamoe_case,
    ),
    CanonicalKernelCase("kda_backward_packed", prepare_native_kda_backward_case),
    CanonicalKernelCase("kda_decode_multishape", prepare_native_kda_decode_case),
    CanonicalKernelCase("msa_decode_multishape", prepare_native_msa_decode_case),
    CanonicalKernelCase("mla_dsv4_multishape", prepare_native_mla_case),
    CanonicalKernelCase(
        "msa_prefill_multishape", prepare_native_msa_case,
    ),
    CanonicalKernelCase(
        "vsa_multishape", prepare_native_vsa_case,
    ),
    CanonicalKernelCase("merge_state", prepare_merge_state_case),
    CanonicalKernelCase(
        "cudnn_sm103_flex_attention_forward",
        prepare_flex_attention_forward_sm103_case,
        racecheck_expected_verdict="review",
        racecheck_review_kinds=frozenset({"tmem_lifetime_review"}),
        racecheck_review_counts=(("tmem_lifetime_review", 2),),
    ),
    CanonicalKernelCase(
        "cudnn_sm100_flex_attention_forward_hd256", prepare_flex_attention_forward_hd256_case
    ),
    CanonicalKernelCase(
        "cudnn_sm100_flex_attention_backward", prepare_flex_attention_backward_case
    ),
    CanonicalKernelCase("act_and_mul", prepare_act_and_mul_case),
    CanonicalKernelCase(
        "silu_and_mul_nvfp4_experts_quantize",
        prepare_silu_nvfp4_experts_case,
    ),
    CanonicalKernelCase(
        "flash_attention4",
        _flash_attention4,
        racecheck_expected_verdict="review",
        racecheck_review_kinds=frozenset({"tmem_lifetime_review"}),
        racecheck_review_counts=(("tmem_lifetime_review", 8),),
    ),
    CanonicalKernelCase(
        "flash_attention_backward_sm100",
        _flash_attention_backward_sm100,
    ),
    CanonicalKernelCase(
        "sm100_fp8_fp4_mega_moe",
        prepare_mega_moe_case,
        # Combine intentionally reuses the shared-pool prefix previously
        # named ``smem_expert_count``. Its raw TMA overwrite has no logical
        # buffer tag, so the later ``combine_chunks`` read retains one
        # aggregated stale-name advisory while physical ordering stays clean.
        racecheck_expected_verdict="review",
        racecheck_review_kinds=frozenset({"alias_stale_read"}),
        racecheck_review_counts=(("alias_stale_read", 1),),
    ),
    CanonicalKernelCase(
        "deepgemm_sm100_fp4_mqa_logits",
        _dense_fp4_mqa,
    ),
    CanonicalKernelCase(
        "deepgemm_sm100_fp8_mqa_logits",
        _dense_fp8_mqa,
    ),
    CanonicalKernelCase(
        "deepgemm_sm100_fp4_paged_mqa_logits",
        lambda: prepare_paged_mqa_case(input_format="fp4"),
        expected_verdict="review",
    ),
    CanonicalKernelCase(
        "deepgemm_sm100_fp8_paged_mqa_logits",
        lambda: prepare_paged_mqa_case(input_format="fp8"),
    ),
    CanonicalKernelCase(
        "deepgemm_sm100_tf32_hc_prenorm_gemm",
        _tf32_hc,
    ),
    CanonicalKernelCase(
        "deepgemm_sm100_fp8_bmm",
        prepare_fp8_bmm_case,
        expected_verdict="review",
        racecheck_review_kinds=frozenset({"uninitialized_read"}),
        racecheck_review_counts=(("uninitialized_read", 224),),
    ),
    CanonicalKernelCase(
        "deepgemm_sm100_k_grouped_fp8_gemm_contiguous",
        prepare_k_grouped_fp8_case,
    ),
    CanonicalKernelCase(
        "deepgemm_sm100_m_grouped_fp8_gemm_contiguous",
        _grouped_fp8,
        expected_verdict="review",
        racecheck_review_kinds=frozenset({"uninitialized_read"}),
        racecheck_review_counts=(("uninitialized_read", 64),),
    ),
    CanonicalKernelCase(
        "deepgemm_sm100_m_grouped_fp8_gemm_masked",
        prepare_m_grouped_masked_fp8_case,
        expected_verdict="review",
        racecheck_review_kinds=frozenset({"uninitialized_read"}),
        racecheck_review_counts=(("uninitialized_read", 192),),
    ),
    CanonicalKernelCase(
        "flash_mla_sparse_fwd",
        _flash_mla_sparse_fwd,
        racecheck_expected_verdict="review",
        racecheck_review_kinds=frozenset({"alias_stale_read"}),
        racecheck_review_counts=(("alias_stale_read", 1),),
    ),
    CanonicalKernelCase(
        "sparse_flashmla_decode_head64",
        prepare_sparse_decode_head64_case,
        racecheck_phase_expected_verdicts=("review", "clean"),
        racecheck_review_kinds=frozenset({"alias_stale_read"}),
        racecheck_phase_review_counts=((("alias_stale_read", 7),), ()),
    ),
    CanonicalKernelCase(
        "sparse_flashmla_prefill_head128_phase1",
        _sparse_prefill_head128,
        racecheck_expected_verdict="review",
        racecheck_review_kinds=frozenset({"alias_stale_read"}),
        racecheck_review_counts=(("alias_stale_read", 2),),
    ),
    CanonicalKernelCase(
        "sparse_flashmla_prefill_head128_small_topk_phase1",
        _sparse_prefill_head128_small_topk,
        racecheck_expected_verdict="review",
        racecheck_review_kinds=frozenset({"alias_stale_read"}),
        racecheck_review_counts=(("alias_stale_read", 1),),
    ),
    CanonicalKernelCase(
        "sparse_flashmla_prefill_head64_phase1",
        _sparse_prefill_head64,
        racecheck_expected_verdict="review",
        racecheck_review_kinds=frozenset({"alias_stale_read"}),
        racecheck_review_counts=(("alias_stale_read", 1),),
    ),
    CanonicalKernelCase("fp16_bf16_gemm", _fp16_gemm),
    CanonicalKernelCase("bmm_fp8_rubin", prepare_bmm_fp8_rubin_case),
    CanonicalKernelCase(
        "cudnn_sm100_dense_blockscaled_gemm_persistent_amax",
        prepare_cudnn_dense_blockscaled_amax_case,
    ),
    CanonicalKernelCase(
        "cudnn_sm100_dense_blockscaled_gemm_persistent_srelu_quant",
        prepare_cudnn_dense_blockscaled_srelu_quant_case,
    ),
    CanonicalKernelCase(
        "cudnn_sm100_dense_blockscaled_gemm_persistent_dsrelu_quant",
        prepare_cudnn_dense_blockscaled_dsrelu_quant_case,
    ),
    CanonicalKernelCase(
        "cudnn_sm100_dense_gemm_persistent_swiglu",
        prepare_cudnn_dense_gemm_persistent_swiglu_case,
    ),
    CanonicalKernelCase(
        "cudnn_sm100_dense_blockscaled_gemm_persistent_swiglu_interleaved_quant",
        prepare_cudnn_dense_blockscaled_swiglu_quant_case,
    ),
    CanonicalKernelCase(
        "cudnn_sm100_moe_blockscaled_grouped_gemm_dglu_dbias",
        prepare_cudnn_moe_dglu_dbias_case,
    ),
    CanonicalKernelCase(
        "cudnn_sm100_moe_grouped_gemm_dglu_dbias",
        prepare_cudnn_moe_bf16_dglu_dbias_case,
    ),
    CanonicalKernelCase(
        "cudnn_sm100_kda_bprop_f16",
        prepare_cudnn_kda_bprop_case,
    ),
    CanonicalKernelCase(
        "cudnn_sm100_gdn_bprop_f16",
        prepare_cudnn_gdn_bprop_case,
        expected_verdict="review",
        synccheck_phase_expected_verdicts=("clean", "review"),
        racecheck_phase_expected_verdicts=("clean", "review"),
        racecheck_review_kinds=frozenset({"tmem_lifetime_review", "uninitialized_read"}),
        racecheck_phase_review_counts=(
            (),
            (("tmem_lifetime_review", 8), ("uninitialized_read", 16384)),
        ),
    ),
    CanonicalKernelCase(
        "cudnn_sm100_gdn2_prefill_f16",
        prepare_cudnn_gdn2_prefill_case,
    ),
    CanonicalKernelCase(
        "cudnn_sm100_gdn2_bprop_f16",
        prepare_cudnn_gdn2_bprop_case,
    ),
    CanonicalKernelCase(
        "cudnn_sm100_gdn_prefill_f16",
        prepare_cudnn_gdn_prefill_case,
    ),
    CanonicalKernelCase(
        "cudnn_sm100_gdn_recompute_f16",
        prepare_cudnn_gdn_recompute_case,
        expected_verdict="review",
        synccheck_phase_expected_verdicts=("clean", "review"),
        racecheck_phase_expected_verdicts=("clean", "review"),
        racecheck_phase_review_counts=((), (("uninitialized_read", 8192),)),
    ),
    CanonicalKernelCase(
        "cudnn_sm100_dsa_sparse_attention_backward",
        prepare_cudnn_dsa_sparse_attention_backward_case,
    ),
    CanonicalKernelCase(
        "cudnn_sm100_bsa_backward_blk128",
        prepare_cudnn_bsa_backward_blk128_case,
    ),
    CanonicalKernelCase(
        "cudnn_sm100_bsa_backward_blk64",
        prepare_cudnn_bsa_backward_blk64_case,
    ),
    CanonicalKernelCase(
        "cudnn_sm100_bsa_forward_blk128",
        prepare_cudnn_bsa_forward_blk128_case,
        racecheck_expected_verdict="review",
        racecheck_review_kinds=frozenset({"tmem_lifetime_review"}),
        racecheck_review_counts=(("tmem_lifetime_review", 2),),
    ),
    CanonicalKernelCase(
        "cudnn_sm100_bsa_forward_blk64",
        prepare_cudnn_bsa_forward_blk64_case,
        racecheck_expected_verdict="review",
        racecheck_review_kinds=frozenset({"tmem_lifetime_review"}),
        racecheck_review_counts=(("tmem_lifetime_review", 2),),
    ),
    CanonicalKernelCase(
        "cudnn_sm100_bsa_forward_combine_blk64",
        prepare_cudnn_bsa_forward_combine_blk64_case,
        expected_verdict="review",
        racecheck_review_counts=(("uninitialized_read", 128),),
    ),
    CanonicalKernelCase(
        "cudnn_sm100_csa_compressor_fwd",
        prepare_cudnn_csa_compressor_fwd_case,
    ),
    CanonicalKernelCase(
        "cudnn_sm100_gemm_proj_rope_mxfp8_bf16in",
        prepare_cudnn_gemm_proj_rope_mxfp8_bf16in_case,
    ),
    CanonicalKernelCase(
        "cudnn_sm100_gemm_proj_rope_mxfp8_mxfp8in",
        prepare_cudnn_gemm_proj_rope_mxfp8_mxfp8in_case,
    ),
    CanonicalKernelCase(
        "deepgemm_sm100_fp8_gemm_1d1d",
        _fp8_blockwise,
        expected_verdict="review",
        racecheck_review_kinds=frozenset({"uninitialized_read"}),
        racecheck_review_counts=(("uninitialized_read", 224),),
    ),
    CanonicalKernelCase(
        "gdn_prefill_sm100",
        prepare_gdn_prefill_case,
        racecheck_expected_verdict="review",
        racecheck_review_kinds=frozenset({"tmem_lifetime_review"}),
        racecheck_review_counts=(("tmem_lifetime_review", 16),),
    ),
    CanonicalKernelCase(
        "gdn_cp_prefill_sm100",
        prepare_gdn_cp_prefill_case,
        racecheck_phase_expected_verdicts=("clean", "review", "clean", "review"),
        racecheck_review_kinds=frozenset({"tmem_lifetime_review"}),
        racecheck_phase_review_counts=(
            (),
            (("tmem_lifetime_review", 12),),
            (),
            (("tmem_lifetime_review", 12),),
        ),
    ),
    CanonicalKernelCase(
        "kda_forward_portfolio_multishape",
        prepare_native_kda_forward_case,
        synccheck_max_diagnostic_bytes=32 * 1024 * 1024,
    ),
    CanonicalKernelCase(
        "recurrent_kda_decode_grouped",
        prepare_recurrent_kda_grouped_case,
        engine_max_workers=1,
    ),
    CanonicalKernelCase(
        "recurrent_kda_decode_one_warp",
        prepare_recurrent_kda_one_warp_case,
    ),
    CanonicalKernelCase(
        "gdn_decode_bf16_ilp4",
        prepare_gdn_decode_ilp4_case,
        engine_max_workers=1,
    ),
    CanonicalKernelCase(
        "gdn_decode_bf16_wide_vec_mtp",
        prepare_gdn_decode_mtp_case,
        engine_max_workers=1,
    ),
    CanonicalKernelCase(
        "gdn_decode_bf16_wide_vec_t1",
        prepare_gdn_decode_t1_case,
        engine_max_workers=1,
    ),
    CanonicalKernelCase(
        "gdn_decode_fp32_mtp_warp",
        prepare_gdn_decode_fp32_mtp_warp_case,
        expected_verdict="review",
        racecheck_review_kinds=frozenset({"uninitialized_read"}),
        # 17,408 physical locations are each consumed by three source statements.
        racecheck_review_counts=(("uninitialized_read", 52224),),
        engine_max_workers=1,
    ),
    CanonicalKernelCase(
        "selective_state_update_stp_simple",
        prepare_selective_state_update_stp_simple_case,
    ),
    CanonicalKernelCase(
        "selective_state_update_stp_vertical",
        prepare_selective_state_update_stp_vertical_case,
    ),
    CanonicalKernelCase(
        "selective_state_update_stp_horizontal",
        prepare_selective_state_update_stp_horizontal_case,
    ),
    CanonicalKernelCase(
        "selective_state_update_mtp_simple",
        prepare_selective_state_update_mtp_simple_case,
    ),
    CanonicalKernelCase(
        "selective_state_update_mtp_vertical",
        prepare_selective_state_update_mtp_vertical_case,
        expected_verdict="review",
        racecheck_review_kinds=frozenset({"uninitialized_read"}),
        racecheck_review_counts=(("uninitialized_read", 2048),),
    ),
    CanonicalKernelCase(
        "selective_state_update_mtp_horizontal",
        prepare_selective_state_update_mtp_horizontal_case,
    ),
    CanonicalKernelCase("nvfp4_gemm", _nvfp4),
    CanonicalKernelCase("mxfp4_quantize", prepare_mxfp4_quantize_case),
    CanonicalKernelCase("mxfp8_quantize", prepare_mxfp8_quantize_case),
    CanonicalKernelCase("nvfp4_quantize", prepare_nvfp4_quantize_case),
    CanonicalKernelCase(
        "nvfp4_quantize_per_token",
        prepare_nvfp4_quantize_per_token_case,
    ),
    CanonicalKernelCase("rmsnorm", prepare_rmsnorm_case),
    CanonicalKernelCase(
        "flashinfer_rmsnorm_fp4quant",
        prepare_flashinfer_rmsnorm_fp4quant_case,
        expected_verdict="review",
        engine_max_workers=1,
    ),
    CanonicalKernelCase(
        "flashinfer_rmsnorm", prepare_flashinfer_rmsnorm_case, expected_verdict="review"
    ),
    CanonicalKernelCase(
        "flashinfer_rmsnorm_quant", prepare_flashinfer_rmsnorm_quant_case, expected_verdict="review"
    ),
    CanonicalKernelCase(
        "flashinfer_fused_add_rmsnorm",
        prepare_flashinfer_fused_add_rmsnorm_case,
        expected_verdict="review",
    ),
    CanonicalKernelCase(
        "flashinfer_fused_add_rmsnorm_quant",
        prepare_flashinfer_fused_add_rmsnorm_quant_case,
        expected_verdict="review",
    ),
    CanonicalKernelCase("flashinfer_layernorm", prepare_flashinfer_layernorm_case),
    CanonicalKernelCase(
        "flashinfer_fused_dit_layernorm", prepare_flashinfer_fused_dit_layernorm_case
    ),
    CanonicalKernelCase(
        "flashinfer_add_rmsnorm_fp4quant",
        prepare_flashinfer_add_rmsnorm_fp4quant_case,
        expected_verdict="review",
    ),
    CanonicalKernelCase(
        "flashinfer_qk_rmsnorm", prepare_flashinfer_qk_rmsnorm_case, expected_verdict="review"
    ),
    CanonicalKernelCase("fast_topk_clusters", prepare_fast_topk_clusters_case),
    CanonicalKernelCase("filtered_topk", prepare_filtered_topk_case),
    CanonicalKernelCase("radix_topk_multi_cta", prepare_radix_topk_multi_cta_case),
    CanonicalKernelCase("radix_topk_single_cta", prepare_radix_topk_single_cta_case),
    CanonicalKernelCase(
        "stable_sort_topk_by_value",
        prepare_stable_sort_topk_by_value_case,
        racecheck_expected_verdict="review",
        racecheck_review_kinds=frozenset({"alias_stale_read"}),
        racecheck_review_counts=(("alias_stale_read", 3),),
    ),
    CanonicalKernelCase("msa_sparse_atten_fwd_sm100", prepare_msa_sparse_atten_fwd_case),
    CanonicalKernelCase(
        "msa_sparse_atten_fwd_combine_sm100",
        prepare_msa_sparse_atten_fwd_combine_case,
        expected_verdict="review",
    ),
    CanonicalKernelCase(
        "msa_sparse_atten_fwd_nvfp4_kv_sm100",
        prepare_msa_sparse_atten_fwd_nvfp4_kv_case,
        racecheck_expected_verdict="review",
        racecheck_review_kinds=frozenset({"tmem_lifetime_review"}),
        racecheck_review_counts=(("tmem_lifetime_review", 2),),
    ),
    CanonicalKernelCase(
        "msa_sparse_prepare_flat_schedule_sm100", prepare_msa_sparse_prepare_flat_schedule_case
    ),
    CanonicalKernelCase(
        "msa_sparse_prepare_fwd_split_atomic_sm100",
        prepare_msa_sparse_prepare_fwd_split_atomic_case,
    ),
    CanonicalKernelCase(
        "cudnn_sm100_gdn2_recompute_f16",
        prepare_cudnn_gdn2_recompute_case,
        racecheck_phase_expected_verdicts=("clean", "review"),
        racecheck_review_kinds=frozenset({"tmem_lifetime_review"}),
        racecheck_phase_review_counts=((), (("tmem_lifetime_review", 2),)),
    ),
    CanonicalKernelCase("fastcu_nvfp4_gemm_gb300", prepare_fastcu_nvfp4_gemm_case),
    CanonicalKernelCase("flash_attention4_fp4", prepare_flash_attention4_fp4_case),
    CanonicalKernelCase("dense_blockscaled_gemm_sm107", prepare_dense_blockscaled_rubin_case),
    CanonicalKernelCase("grouped_gemm_masked_rubin", prepare_grouped_masked_rubin_case),
    CanonicalKernelCase(
        "blockscaled_contiguous_gather_grouped_gemm_swiglu_fusion_rubin",
        prepare_gather_swiglu_rubin_case,
    ),
)

CANONICAL_KERNEL_MANIFEST = CANONICAL_KERNEL_CASES


MULTI_GPU_ONLY_CANONICAL_KERNELS = frozenset(
    {"allgather_gemm", "gemm_reduce_scatter", "deepep_dispatch", "deepep_combine"}
)


__all__ = [
    "CANONICAL_KERNEL_CASES",
    "CANONICAL_KERNEL_MANIFEST",
    "MULTI_GPU_ONLY_CANONICAL_KERNELS",
    "CanonicalKernelCase",
]
