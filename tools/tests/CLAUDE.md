# Test Instructions

## Running the Test Suites

A fresh worktree first needs the Rust frontend dependency:

```bash
git submodule update --init thirdparty/tvm-rust-ext  # repo root
```

Tests load canonical kernels from the environment's installed `tirx-kernels`
package. Then from `tools/`:

```bash
python -m pip install --no-deps --no-build-isolation ..
(cd src/tirx_harness/numsim/engine-rs && cargo build --all-features)  # engine changes
python -m pytest -q -n 16 --dist=worksteal
```

Always pass `-n`, including for targeted reruns.

Live GPU microtests run by default. `--no-run-numsim-gpu` skips them, but it is
declared in `tests/numsim/conftest.py` and is therefore only accepted when the
invocation targets that directory — a full run rejects it as an unrecognized
argument. To drop them from a full run, deselect by marker instead:

```bash
python -m pytest -q -n 16 --dist=worksteal -m "not numsim_gpu"
```

## Judging a Full-Suite Run

Wiki cases can fail on a clean checkout. A failure count proves nothing on its
own; only the **difference** between failure sets does.

1. Create a pristine worktree at the base commit and initialize its submodules
   (`git submodule update --init --recursive`). A submodule left at a different
   commit manufactures dozens of phantom differences.
2. Run the identical command in both, **serially**. Concurrent runs contend for
   CPU and GPUs and produce failures that reproduce in neither tree alone.
3. Compare the sorted `FAILED`/`ERROR` node id sets. Only ids appearing solely
   in your run are yours; investigate each before attributing it to the
   environment.

A diff cannot detect a false negative: a check your change silently stopped
performing still passes. Whenever a change relaxes or reroutes a check, add a
positive-control test that fails without the fix.

## Performance Thresholds

Performance regressions use the `performance` marker. They stay in the
default suite and intentionally follow the requested pytest `-n` concurrency;
do not add an xdist group or suite-exclusive lock for them.

Every change that can affect performance must rerun the relevant performance
tests and review whether each associated threshold can be lowered. Do not keep
a looser historical threshold merely because the test still passes. Set the
threshold from the current measured maximum plus a justified, small noise
margin, and record the measured maximum, workload/concurrency configuration,
date, and rounding rationale next to the threshold. Any threshold increase
must likewise be supported by current measurements and an explanation.

Before running any performance test, inspect and record the host CPU count,
load average, and instantaneous CPU idle/pressure. Do not start or use a
performance measurement while the host is materially busy; wait for load to
subside and recheck first. Never raise a threshold from a run that skipped this
preflight or ran under load.
