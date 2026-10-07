# Compiler analysis

This directory provides compiler analysis in the installable `tirx-harness` package. Runtime
code lives under `src/tirx_harness/`. Top-level tests and the NumSim test suites
stay outside the wheel; they may load repository fixtures directly.

Keep public imports rooted at `tirx_harness`. Do not add dependencies on the
parent repository layout. The TIRx-enabled `tvm` package is supplied by the
runtime environment and must not be replaced with an unrelated PyPI TVM build.
Native checker code is owned by `src/tirx_harness/numsim/` and follows that
directory's guide.

Validate changes with:

```bash
python -m pip install --no-deps --no-build-isolation ..
python -m pytest -q -n 16
```

[tests/CLAUDE.md](tests/CLAUDE.md) covers the prerequisites a fresh worktree
needs, the GPU and marker options, and how to judge a full-suite result against
a baseline.
