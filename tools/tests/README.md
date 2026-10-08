# Test layout

`tools/tests/` contains the Python/Pytest integration and corpus tests for
the installed `tirx_harness` package.

The embedded NumSim Rust crate follows Cargo's conventional split:

- private unit tests stay beside the implementation in
  `src/tirx_harness/numsim/engine-rs/src/` under `#[cfg(test)]`;
- tests that exercise only the public crate ABI live in
  `src/tirx_harness/numsim/engine-rs/tests/`.

Do not move private Rust unit tests into this directory. Cargo would not
discover them here, and moving them into the crate's integration-test directory
would require making implementation details public solely for testing.
