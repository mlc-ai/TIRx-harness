# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors

"""TIRx port of DeepGEMM's MegaMoE kernel.

Submodules: :mod:`spec` (config, layout, heuristics, compile/launch plumbing),
:mod:`data` (case construction, references, distributed bench harness), and
:mod:`kernel` (the ``PrimFunc`` body).  The public kernel module
:mod:`tirx_kernels.ported.deepgemm.sm100_fp8_fp4_mega_moe` pins the registry surface and
the test/benchmark matrices.
"""
