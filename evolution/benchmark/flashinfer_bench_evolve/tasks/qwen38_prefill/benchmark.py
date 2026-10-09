"""Qwen3.8-27B prefill: one prepared complete-model forward."""

from ..qwen38 import benchmark as shared
from ..qwen38.benchmark import HARNESS_FILES, default_config, tirx_prepare, tirx_run

TASK_NAME = "qwen38_prefill"


def make_workloads(config=None):
    return shared.make_workloads("prefill", config)


def run_suite(config=None, candidate_fn=None, workloads=None, candidate_prepare_fn=None):
    return shared.run_suite(
        "prefill", config, candidate_fn, workloads, candidate_prepare_fn
    )
