"""Kernels ported to TIRx from other libraries, laid out as in ``tirx_kernels.ported``.

``ported/<origin>/<kernel>.py`` keeps the ``tirx_kernels`` kernel protocol
(``KERNEL_META``, ``CONFIGS``, ``get_kernel``, ``run_test``, ``run_bench``) so a
port can move to mlc-ai/tirx-kernels unchanged. Kernel modules must not import
``tirx_harness``; the NumSim bindings for them live under ``tests/``.
"""
