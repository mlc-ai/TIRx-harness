"""Ports of CUTLASS CuTeDSL kernels.

``sm100_all_gather_gemm`` ports the GEMM of
``examples/python/CuTeDSL/cute/blackwell/kernel/distributed/distributed_all_gather_gemm_blackwell.py``
(CUTLASS 0b55a2f691d6) and consumes the gathered shards the way FlashInfer's
cake all-gather matmul does (FlashInfer 776939f).
"""
