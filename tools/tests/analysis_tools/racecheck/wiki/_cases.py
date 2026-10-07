"""CUDA-free inputs for native Racecheck over the established wiki-kernel corpus."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Callable

from tvm import tirx

from tests.numsim.corpus.kernels.attention import (
    FLASH_ATTENTION_BACKWARD_CONFIGS,
    FLASH_ATTENTION4_CONFIGS,
    prepare_flash_attention_backward_case,
    prepare_flash_attention_backward_persistent_analysis_case,
)
from tests.numsim.corpus.kernels.deepgemm import (
    FP4_MQA_CONFIGS,
    FP8_MQA_CONFIGS,
    TF32_HC_CONFIGS,
    prepare_fp4_mqa_case,
    prepare_fp8_mqa_case,
    prepare_tf32_hc_case,
)
from tests.numsim.corpus.kernels.gemm import (
    FP16_BF16_CONFIGS,
    FP8_BLOCKWISE_CONFIGS,
    NVFP4_CONFIGS,
    prepare_fp16_bf16_case,
    prepare_nvfp4_case,
    prepare_fp8_blockwise_case,
)
from tests.numsim.corpus.kernels.attention import prepare_flash_attention4_case
from tirx_harness.numsim.bindings import _tensor_map_base_array


@dataclass(frozen=True)
class WikiRacecheckSpec:
    family: str
    config: dict[str, Any]
    prepare: Callable[[dict[str, Any]], "PreparedWikiRacecheckCase"]
    expected_uninitialized_read: "WikiUninitializedReadExpectation | None" = None

    @property
    def case_id(self) -> str:
        return f"{self.family}__{self.config['label']}"


@dataclass(frozen=True)
class WikiUninitializedReadExpectation:
    count: int
    source_suffix: str
    space: str = "shared"
    allow_spanless_source: bool = False


@dataclass
class PreparedWikiRacecheckCase:
    kernel: Any
    args: dict[str, Any]


def _kwargs(config: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in config.items() if key != "label"}


def _prepare_fp16_bf16(config: dict[str, Any]) -> PreparedWikiRacecheckCase:
    case = prepare_fp16_bf16_case(**_kwargs(config))
    return PreparedWikiRacecheckCase(kernel=case.kernel, args=case.args)


def _prepare_fp8_blockwise(config: dict[str, Any]) -> PreparedWikiRacecheckCase:
    case = prepare_fp8_blockwise_case(**_kwargs(config))
    return PreparedWikiRacecheckCase(kernel=case.kernel, args=case.args)


def _prepare_nvfp4(config: dict[str, Any]) -> PreparedWikiRacecheckCase:
    case = prepare_nvfp4_case(**_kwargs(config))
    return PreparedWikiRacecheckCase(kernel=case.kernel, args=case.args)


def _prepare_flash_attention4(config: dict[str, Any]) -> PreparedWikiRacecheckCase:
    case = prepare_flash_attention4_case(**_kwargs(config))
    return PreparedWikiRacecheckCase(kernel=case.kernel, args=case.args)


def _prepare_flash_attention_backward(
    config: dict[str, Any],
) -> PreparedWikiRacecheckCase:
    return _prepared_numsim_case(prepare_flash_attention_backward_case(**_kwargs(config)))


def _prepare_flash_attention_backward_persistent(
    _config: dict[str, Any],
) -> PreparedWikiRacecheckCase:
    return _prepared_numsim_case(prepare_flash_attention_backward_persistent_analysis_case())


_MQA_RUNTIME_SCALARS = frozenset({"seq_len", "seq_len_kv", "max_seqlen_k", "logits_stride"})


def _prepared_numsim_case(case) -> PreparedWikiRacecheckCase:
    substitutions = {
        param: int(case.args[param.name])
        for param in case.kernel.params
        if param.name in _MQA_RUNTIME_SCALARS
    }
    case.kernel = case.kernel.specialize(substitutions)
    args = {name: value for name, value in case.args.items() if name not in _MQA_RUNTIME_SCALARS}
    tensor_map_buffers = {
        "tensor_map_q": "q_gmem",
        "tensor_map_sf_q": "sf_q_gmem",
        "tensor_map_kv": "kv_gmem",
        "tensor_map_sf_kv": "sf_kv_gmem",
        "tensor_map_kv_scales": "kv_scales_gmem",
        "tensor_map_weights": "weights_gmem",
    }
    for tensor_map_name, buffer_name in tensor_map_buffers.items():
        tensor_map = args.pop(tensor_map_name, None)
        if tensor_map is not None:
            array = _tensor_map_base_array(tensor_map)
            if buffer_name == "q_gmem":
                array = array.reshape(-1, array.shape[-1])
            expected = next(
                buffer
                for buffer in case.kernel.params
                if tirx.is_buffer_var(buffer)
                if str(buffer.name) == buffer_name
            )
            expected_shape = tuple(int(extent) for extent in expected.shape)
            if array.shape != expected_shape and array.size == math.prod(expected_shape):
                array = array.reshape(expected_shape)
            args[buffer_name] = array
    return PreparedWikiRacecheckCase(kernel=case.kernel, args=args)


def _prepare_fp4_mqa(config: dict[str, Any]) -> PreparedWikiRacecheckCase:
    return _prepared_numsim_case(prepare_fp4_mqa_case(**_kwargs(config)))


def _prepare_fp8_mqa(config: dict[str, Any]) -> PreparedWikiRacecheckCase:
    return _prepared_numsim_case(prepare_fp8_mqa_case(**_kwargs(config)))


def _prepare_tf32_hc(config: dict[str, Any]) -> PreparedWikiRacecheckCase:
    return _prepared_numsim_case(prepare_tf32_hc_case(**_kwargs(config)))


FLASH_ATTENTION_BACKWARD_PERSISTENT_CONFIGS = ({"label": "s256_h2_persistent_reuse"},)

_DEEPGEMM_1D1D_KERNEL_SOURCE_SUFFIX = "/tirx_kernels/ported/deepgemm/_sm100_fp8_fp4_gemm_1d1d/kernel.py"


_EXPECTED_UNINITIALIZED_READS = {
    (
        "deepgemm_sm100_fp8_gemm_1d1d",
        "swap_ab_m16_n256_k512",
    ): WikiUninitializedReadExpectation(
        count=224,
        source_suffix=_DEEPGEMM_1D1D_KERNEL_SOURCE_SUFFIX,
        allow_spanless_source=True,
    ),
    (
        "deepgemm_sm100_fp8_gemm_1d1d",
        "direct_ab_m512_n608_k512",
    ): WikiUninitializedReadExpectation(
        count=192,
        source_suffix=_DEEPGEMM_1D1D_KERNEL_SOURCE_SUFFIX,
        allow_spanless_source=True,
    ),
}


def _wiki_racecheck_spec(
    family: str,
    config: dict[str, Any],
    prepare: Callable[[dict[str, Any]], PreparedWikiRacecheckCase],
) -> WikiRacecheckSpec:
    copied = dict(config)
    return WikiRacecheckSpec(
        family,
        copied,
        prepare,
        expected_uninitialized_read=_EXPECTED_UNINITIALIZED_READS.get((family, copied["label"])),
    )


WIKI_RACECHECK_SPECS = tuple(
    _wiki_racecheck_spec(family, config, prepare)
    for family, configs, prepare in (
        ("fp16_bf16_gemm", FP16_BF16_CONFIGS, _prepare_fp16_bf16),
        (
            "deepgemm_sm100_fp8_gemm_1d1d",
            FP8_BLOCKWISE_CONFIGS,
            _prepare_fp8_blockwise,
        ),
        ("nvfp4_gemm", NVFP4_CONFIGS, _prepare_nvfp4),
        ("flash_attention4", FLASH_ATTENTION4_CONFIGS, _prepare_flash_attention4),
        (
            "flash_attention_backward_sm100",
            FLASH_ATTENTION_BACKWARD_CONFIGS[:1],
            _prepare_flash_attention_backward,
        ),
        (
            "flash_attention_backward_sm100",
            FLASH_ATTENTION_BACKWARD_PERSISTENT_CONFIGS,
            _prepare_flash_attention_backward_persistent,
        ),
        ("deepgemm_fp4_mqa", FP4_MQA_CONFIGS, _prepare_fp4_mqa),
        ("deepgemm_fp8_mqa", FP8_MQA_CONFIGS, _prepare_fp8_mqa),
        ("deepgemm_tf32_hc", TF32_HC_CONFIGS, _prepare_tf32_hc),
    )
    for config in configs
)


def prepare_wiki_racecheck_case(spec: WikiRacecheckSpec) -> PreparedWikiRacecheckCase:
    return spec.prepare(spec.config)


__all__ = [
    "PreparedWikiRacecheckCase",
    "WIKI_RACECHECK_SPECS",
    "WikiRacecheckSpec",
    "WikiUninitializedReadExpectation",
    "prepare_wiki_racecheck_case",
]
