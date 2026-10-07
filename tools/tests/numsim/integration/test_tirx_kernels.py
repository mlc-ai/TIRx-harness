from __future__ import annotations

import multiprocessing
from concurrent.futures import ProcessPoolExecutor

import pytest

from tests.numsim.support._tirx_kernels import load_tirx_kernel


def _spawn_load(name: str) -> tuple[str, str]:
    module = load_tirx_kernel(name)
    return module.__name__, module.KERNEL_META["name"]


def test_canonical_imports_use_the_real_package_namespace() -> None:
    gemm = load_tirx_kernel("fp16_bf16_gemm")
    mqa = load_tirx_kernel("deepgemm_sm100_fp8_mqa_logits")

    assert gemm.__name__ == "tirx_kernels.gemm.fp16_bf16_gemm"
    assert mqa.__name__ == "tirx_kernels.ported.deepgemm.mqa_logits_fp8"
    assert callable(gemm.get_kernel)
    assert callable(mqa.get_kernel)
    assert load_tirx_kernel("fp16_bf16_gemm") is gemm


@pytest.mark.parametrize(
    ("name", "error"),
    [
        ("not-a-kernel", ValueError),
        ("missing_kernel", KeyError),
    ],
)
def test_rejects_invalid_or_missing_modules(name, error) -> None:
    with pytest.raises(error):
        load_tirx_kernel(name)


def test_spawn_worker_imports_the_installed_package() -> None:
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=1, mp_context=context) as executor:
        module_name, kernel_name = executor.submit(_spawn_load, "flash_attention4").result()

    assert module_name == "tirx_kernels.ported.flashattention.flash_attention4"
    assert kernel_name == "flash_attention4"
