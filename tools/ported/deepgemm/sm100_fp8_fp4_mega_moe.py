# This file is a TIRx port of code from DeepGEMM
# (https://github.com/deepseek-ai/DeepGEMM @ 559d79fb), Copyright (c) 2025 DeepSeek
# SPDX-License-Identifier: Apache-2.0 AND MIT
# SPDX-FileCopyrightText: Copyright TIRx authors

"""Fused MoE megakernel (MegaMoE) -- fp8 activations, fp4 weights, SM100.

Reference: ``deep_gemm.fp8_fp4_mega_moe``.  The implementation lives in
:mod:`tirx_kernels.ported.deepgemm._sm100_fp8_fp4_mega_moe`; this module pins the test
and benchmark matrices and the registry surface.

Upstream sources: deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh, csrc/apis/mega.h,
csrc/jit_kernels/heuristics/mega_moe.h.
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from unittest import SkipTest

from ._sm100_fp8_fp4_mega_moe.data import _run_distributed
from ._sm100_fp8_fp4_mega_moe.kernel import get_kernel as get_kernel
from ._sm100_fp8_fp4_mega_moe.spec import (
    _PREPARED_LIBRARY_ENV,
    MegaMoeConfig,
    _compile_tirx_mega_moe_for_config,
    _get_mega_moe_cuda_compile_mode,
)
from ._sm100_fp8_fp4_mega_moe.spec import fp8_fp4_mega_moe as fp8_fp4_mega_moe
from ._sm100_fp8_fp4_mega_moe.spec import (
    launch_prepared_tirx_fp8_fp4_mega_moe as launch_prepared_tirx_fp8_fp4_mega_moe,
)
from ._sm100_fp8_fp4_mega_moe.spec import (
    prepare_tirx_fp8_fp4_mega_moe as prepare_tirx_fp8_fp4_mega_moe,
)

KERNEL_META = {
    "name": "sm100_fp8_fp4_mega_moe",
    "category": "ported.deepgemm",
    "runtime_cuda_archs": ["sm_100a", "sm_103a", "sm_107a"],
    "reference_requirements": (
        {
            "package": "deep-gemm",
            "git": {
                "url": "https://github.com/deepseek-ai/DeepGEMM.git",
                "commit": "559d79fb6994a58b8a15b4b93bf13ccc16edf247",
            },
            "import": "deep_gemm",
        },
    ),
}


@dataclass
class PreparedMegaMoeBench:
    config: MegaMoeConfig
    temporary_directory: Any
    executables: tuple[Any, Any]
    library_paths: dict[bool, Path]

    @property
    def required_num_gpus(self) -> int:
        return self.config.num_processes

    def run_gpu(
        self,
        *,
        warmup: int | None = None,
        repeat: int | None = None,
        timer: str | None = None,
        rounds: int = 1,
        cooldown_s: float = 1.0,
    ) -> dict[str, Any]:
        from tirx_kernels.runner import current_cuda_assignment

        device_indices, device_uuids = current_cuda_assignment()
        if len(device_indices) != self.config.num_processes:
            raise RuntimeError(
                f"MegaMoE requires {self.config.num_processes} GPUs, "
                f"assignment has {len(device_indices)}"
            )
        assignments = {
            _PREPARED_LIBRARY_ENV[collect_stats]: str(path)
            for collect_stats, path in self.library_paths.items()
        }
        previous = {name: os.environ.get(name) for name in assignments}
        os.environ.update(assignments)
        try:
            return _run_distributed(
                self.config,
                "bench",
                warmup=warmup,
                repeat=repeat,
                timer=timer,
                rounds=rounds,
                cooldown_s=cooldown_s,
                device_indices=device_indices,
                device_uuids=device_uuids,
            )
        finally:
            for name, value in previous.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value

    def close(self) -> None:
        self.temporary_directory.cleanup()


def run_gpu(prepared: PreparedMegaMoeBench, **kwargs: Any) -> dict[str, Any]:
    """Run the prepared distributed MegaMoE stage after GPU assignment."""
    return prepared.run_gpu(**kwargs)


def _case(
    label: str,
    *,
    tok: int,
    h: int,
    i: int,
    e: int,
    k: int,
    g: int = 1,
    max_tok: int | None = None,
    s: int = 0,
) -> dict:
    """One config: `g` ranks routing `tok` tokens over `e` experts, top-`k`.

    `max_tok` is the per-rank capacity when it differs from the token count.
    `s` is the shared-expert count: a width multiplier adding one fused
    rank-local FFN of intermediate width `s * i` whose unweighted output folds
    into every token's combine sum.
    Every case runs with the upstream default clamp and fast-math setting.
    """
    return {
        "num_processes": g,
        "num_max_tokens_per_rank": tok if max_tok is None else max_tok,
        "num_tokens": tok,
        "hidden": h,
        "intermediate_hidden": i,
        "num_experts": e,
        "num_topk": k,
        "num_shared_experts": s,
        "activation_clamp": 10.0,
        "fast_math": 1,
        "label": label,
    }


# The first nine cases cover one block_m bucket each from
# `_get_block_config_for_mega_moe` (16, 32, 64, 96, 128, 192), so a per-PR
# sm100a run exercises every heuristic-selected path; each token count sits
# just below its bucket boundary. The rest sweep production shapes.
CONFIGS = [
    _case("p1_tok2_h1024_i512_e2_k1_bm16", tok=2, max_tok=4, h=1024, i=512, e=2, k=1),
    _case("p2_tok2_h1024_i512_e4_k1_bm16", g=2, tok=2, max_tok=4, h=1024, i=512, e=4, k=1),
    _case("p4_tok2_h1024_i512_e8_k1_bm16", g=4, tok=2, max_tok=4, h=1024, i=512, e=8, k=1),
    _case("p6_tok2_h1024_i512_e12_k1_bm16", g=6, tok=2, max_tok=4, h=1024, i=512, e=12, k=1),
    _case("p1_tok16_h1024_i512_e2_k2_bm32", tok=16, h=1024, i=512, e=2, k=2),
    _case("p1_tok32_h1024_i512_e2_k2_bm64", tok=32, h=1024, i=512, e=2, k=2),
    _case("p1_tok64_h1024_i512_e2_k2_bm96", tok=64, h=1024, i=512, e=2, k=2),
    _case("p1_tok96_h1024_i512_e2_k2_bm128", tok=96, h=1024, i=512, e=2, k=2),
    _case("p1_tok192_h1024_i512_e2_k2_bm192", tok=192, h=1024, i=512, e=2, k=2),
    _case("t64_m64_h7168_i3072_e384_k6_g1", tok=64, h=7168, i=3072, e=384, k=6),
    _case("t64_m64_h7168_i3072_e384_k6_g2", g=2, tok=64, h=7168, i=3072, e=384, k=6),
    _case("t8192_m8192_h7168_i3072_e384_k6_g2", g=2, tok=8192, h=7168, i=3072, e=384, k=6),
    _case("t64_m64_h7168_i3072_e384_k6_g4", g=4, tok=64, h=7168, i=3072, e=384, k=6),
    _case("t8192_m8192_h7168_i3072_e384_k6_g4", g=4, tok=8192, h=7168, i=3072, e=384, k=6),
    _case("t64_m64_h7168_i3072_e384_k6_g6", g=6, tok=64, h=7168, i=3072, e=384, k=6),
    _case("t8_h1024_i512_e24_k2_g1", tok=8, h=1024, i=512, e=24, k=2),
    _case("t16_h2048_i1024_e48_k2_g1", tok=16, h=2048, i=1024, e=48, k=2),
    _case("t64_h4096_i1536_e96_k4_g1", tok=64, h=4096, i=1536, e=96, k=4),
    _case("t32_m64_h7168_i3072_e384_k6_g1", tok=32, max_tok=64, h=7168, i=3072, e=384, k=6),
    _case("t8192_h7168_i3072_e64_k6_g1", tok=8192, h=7168, i=3072, e=64, k=6),
    _case("t8192_h7168_i3072_e192_k6_g1", tok=8192, h=7168, i=3072, e=192, k=6),
    _case("t8192_m8192_h7168_i3072_e384_k6_g1", tok=8192, h=7168, i=3072, e=384, k=6),
    _case("t256_h7168_i3072_e384_k6_g4", g=4, tok=256, h=7168, i=3072, e=384, k=6),
    _case("t1024_h7168_i3072_e384_k6_g4", g=4, tok=1024, h=7168, i=3072, e=384, k=6),
    _case("t8192_m8192_h7168_i3072_e384_k6_g6", g=6, tok=8192, h=7168, i=3072, e=384, k=6),
    # Shared experts (`s > 0`). The block-config heuristic has no `S` term, so
    # each case lands in the same block_m bucket as its `s=0` twin; together
    # these re-cover all six buckets with the shared path enabled.
    _case("p1_tok2_h1024_i512_e2_k1_bm16_s1", tok=2, max_tok=4, h=1024, i=512, e=2, k=1, s=1),
    _case("p1_tok2_h1024_i512_e2_k1_bm16_s2", tok=2, max_tok=4, h=1024, i=512, e=2, k=1, s=2),
    _case("p1_tok16_h1024_i512_e2_k2_bm32_s1", tok=16, h=1024, i=512, e=2, k=2, s=1),
    _case("p1_tok32_h1024_i512_e2_k2_bm64_s1", tok=32, h=1024, i=512, e=2, k=2, s=1),
    _case("p1_tok64_h1024_i512_e2_k2_bm96_s1", tok=64, h=1024, i=512, e=2, k=2, s=1),
    _case("p1_tok96_h1024_i512_e2_k2_bm128_s1", tok=96, h=1024, i=512, e=2, k=2, s=1),
    _case("p1_tok192_h1024_i512_e2_k2_bm192_s1", tok=192, h=1024, i=512, e=2, k=2, s=1),
    _case("p2_tok2_h1024_i512_e4_k1_bm16_s1", g=2, tok=2, max_tok=4, h=1024, i=512, e=4, k=1, s=1),
    _case("t64_m64_h7168_i3072_e384_k6_g1_s1", tok=64, h=7168, i=3072, e=384, k=6, s=1),
    _case("t8192_m8192_h7168_i3072_e384_k6_g1_s1", tok=8192, h=7168, i=3072, e=384, k=6, s=1),
    _case("t8192_m8192_h7168_i3072_e384_k6_g2_s1", g=2, tok=8192, h=7168, i=3072, e=384, k=6, s=1),
    _case("t8192_m8192_h7168_i3072_e384_k6_g4_s1", g=4, tok=8192, h=7168, i=3072, e=384, k=6, s=1),
    _case("t8192_m8192_h7168_i3072_e384_k6_g6_s1", g=6, tok=8192, h=7168, i=3072, e=384, k=6, s=1),
]


def _make_config(
    num_processes=1,
    num_max_tokens_per_rank=4,
    num_tokens=2,
    hidden=1024,
    intermediate_hidden=512,
    num_experts=2,
    num_topk=1,
    num_shared_experts=0,
    activation_clamp=10.0,
    fast_math=1,
) -> MegaMoeConfig:
    config = MegaMoeConfig(
        num_processes=num_processes,
        num_max_tokens_per_rank=num_max_tokens_per_rank,
        num_tokens=num_tokens,
        hidden=hidden,
        intermediate_hidden=intermediate_hidden,
        num_experts=num_experts,
        num_topk=num_topk,
        num_shared_experts=num_shared_experts,
        activation_clamp=activation_clamp,
        fast_math=fast_math,
    )
    config.validate()
    return config


def prepare_data(
    num_processes=1,
    num_max_tokens_per_rank=4,
    num_tokens=2,
    hidden=1024,
    intermediate_hidden=512,
    num_experts=2,
    num_topk=1,
    num_shared_experts=0,
    activation_clamp=10.0,
    fast_math=1,
) -> dict[str, Any]:
    return {
        "config": _make_config(
            num_processes=num_processes,
            num_max_tokens_per_rank=num_max_tokens_per_rank,
            num_tokens=num_tokens,
            hidden=hidden,
            intermediate_hidden=intermediate_hidden,
            num_experts=num_experts,
            num_topk=num_topk,
            num_shared_experts=num_shared_experts,
            activation_clamp=activation_clamp,
            fast_math=fast_math,
        )
    }


def _assert_correctness_result(result: dict[str, Any]) -> None:
    if result["status"] == "SKIP":
        raise SkipTest(
            f"{result['reason']} DeepGEMM reference source={result['reference_source']} "
            f"checksum={result['reference_checksum']:.4f}"
        )
    assert result["deepgemm_max_abs_diff"] == 0.0, (
        f"Expected bitwise parity with DeepGEMM reference, got deepgemm_max_abs_diff="
        f"{result['deepgemm_max_abs_diff']}"
    )
    assert result["stats_max_abs_diff"] == 0, (
        "Expected cumulative_local_expert_recv_stats parity with DeepGEMM, got "
        f"stats_max_abs_diff={result['stats_max_abs_diff']}"
    )


def check_correctness(
    outputs: dict[str, Any],
    num_processes=1,
    num_max_tokens_per_rank=4,
    num_tokens=2,
    hidden=1024,
    intermediate_hidden=512,
    num_experts=2,
    num_topk=1,
    num_shared_experts=0,
    activation_clamp=10.0,
    fast_math=1,
) -> None:
    result = outputs.get("result")
    if result is None:
        result = _run_distributed(
            _make_config(
                num_processes=num_processes,
                num_max_tokens_per_rank=num_max_tokens_per_rank,
                num_tokens=num_tokens,
                hidden=hidden,
                intermediate_hidden=intermediate_hidden,
                num_experts=num_experts,
                num_topk=num_topk,
                num_shared_experts=num_shared_experts,
                activation_clamp=activation_clamp,
                fast_math=fast_math,
            ),
            "test",
        )
    _assert_correctness_result(result)


def run_test(
    num_processes=1,
    num_max_tokens_per_rank=4,
    num_tokens=2,
    hidden=1024,
    intermediate_hidden=512,
    num_experts=2,
    num_topk=1,
    num_shared_experts=0,
    activation_clamp=10.0,
    fast_math=1,
):
    config = _make_config(
        num_processes=num_processes,
        num_max_tokens_per_rank=num_max_tokens_per_rank,
        num_tokens=num_tokens,
        hidden=hidden,
        intermediate_hidden=intermediate_hidden,
        num_experts=num_experts,
        num_topk=num_topk,
        num_shared_experts=num_shared_experts,
        activation_clamp=activation_clamp,
        fast_math=fast_math,
    )
    result = _run_distributed(config, "test")
    _assert_correctness_result(result)


def run_bench(
    num_processes=1,
    num_max_tokens_per_rank=128,
    num_tokens=96,
    hidden=1024,
    intermediate_hidden=512,
    num_experts=8,
    num_topk=2,
    num_shared_experts=0,
    activation_clamp=10.0,
    fast_math=1,
    *,
    warmup=None,
    repeat=None,
    timer=None,
    **kwargs,
):
    config = _make_config(
        num_processes=num_processes,
        num_max_tokens_per_rank=num_max_tokens_per_rank,
        num_tokens=num_tokens,
        hidden=hidden,
        intermediate_hidden=intermediate_hidden,
        num_experts=num_experts,
        num_topk=num_topk,
        num_shared_experts=num_shared_experts,
        activation_clamp=activation_clamp,
        fast_math=fast_math,
    )
    return prepare_bench(**asdict(config)).run_gpu(
        warmup=warmup,
        repeat=repeat,
        timer=timer,
        rounds=int(kwargs.pop("rounds", 1)),
        cooldown_s=float(kwargs.pop("cooldown_s", 1.0)),
    )


def prepare_bench(
    num_processes=1,
    num_max_tokens_per_rank=128,
    num_tokens=96,
    hidden=1024,
    intermediate_hidden=512,
    num_experts=8,
    num_topk=2,
    num_shared_experts=0,
    activation_clamp=10.0,
    fast_math=1,
):
    """Compile both legal benchmark specializations before GPU assignment."""
    from tirx_kernels.runner import prepared_gpu_benchmark

    config = _make_config(
        num_processes=num_processes,
        num_max_tokens_per_rank=num_max_tokens_per_rank,
        num_tokens=num_tokens,
        hidden=hidden,
        intermediate_hidden=intermediate_hidden,
        num_experts=num_experts,
        num_topk=num_topk,
        num_shared_experts=num_shared_experts,
        activation_clamp=activation_clamp,
        fast_math=fast_math,
    )
    temporary_directory = tempfile.TemporaryDirectory(prefix="tirx-megamoe-prepared-")
    library_paths: dict[bool, Path] = {}
    executables = []
    try:
        for collect_stats in (False, True):
            executable = _compile_tirx_mega_moe_for_config(
                **asdict(config),
                collect_stats=collect_stats,
                cuda_compile_mode=_get_mega_moe_cuda_compile_mode(),
            )
            library_path = Path(temporary_directory.name) / (
                "mega_moe_stats.so" if collect_stats else "mega_moe_no_stats.so"
            )
            executable.export_library(str(library_path))
            executables.append(executable)
            library_paths[collect_stats] = library_path
    except BaseException:
        temporary_directory.cleanup()
        raise
    state = PreparedMegaMoeBench(
        config=config,
        temporary_directory=temporary_directory,
        executables=tuple(executables),
        library_paths=library_paths,
    )
    return prepared_gpu_benchmark(
        run_gpu, state, required_num_gpus=config.num_processes, close=state.close
    )
