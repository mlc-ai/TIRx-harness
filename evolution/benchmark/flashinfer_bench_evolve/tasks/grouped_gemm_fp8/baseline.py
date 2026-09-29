"""Reuse the tirx-kernels input preparation and DeepGEMM baseline."""

import torch


def prepare_data(config, device):
    from tirx_kernels.ported.deepgemm._sm100_fp8_fp4_gemm_1d1d.data import (
        prepare_m_grouped_contiguous,
    )
    from tirx_kernels.ported.deepgemm._sm100_fp8_fp4_gemm_1d1d.spec import align_up, make_actual_ms

    with torch.cuda.device(device):
        data = prepare_m_grouped_contiguous(**config)
    data["actual_ms"] = tuple(
        make_actual_ms(config["num_groups"], config["expected_m_per_group"], config["seed"])
    )
    data["aligned_ms"] = tuple(align_up(m, data["alignment"]) for m in data["actual_ms"])
    return data


def output_views(output, actual_ms, aligned_ms):
    # Exclude undefined padding from the shared poisoning and correctness checks.
    views, start = [], 0
    for actual, aligned in zip(actual_ms, aligned_ms, strict=True):
        views.append(output.narrow(0, start, actual))
        start += aligned
    return tuple(views)


def prepare(config, device):
    from tirx_kernels.ported.deepgemm._sm100_fp8_fp4_gemm_1d1d.data import (
        deepgemm_launch_m_grouped_contiguous,
    )

    data = prepare_data(config, device)
    launch, output = deepgemm_launch_m_grouped_contiguous(data, out=data["d"])
    return launch, output_views(output, data["actual_ms"], data["aligned_ms"])


def run_prepared(launch, outputs):
    launch()
    return outputs


def run(config, device):
    return run_prepared(*prepare(config, device))
