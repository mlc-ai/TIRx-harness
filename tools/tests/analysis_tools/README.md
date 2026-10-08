# Native analysis tool tests

This tree owns integration coverage added specifically for the public
Synccheck and Racecheck tools. Tests that exercise a single tool live in that
tool's directory. Shared engine, protocol, and analysis-facade
coverage lives in `shared/`.

Generic NumSim ABI, frontend, runtime, and code-generation tests remain under
`tests/numsim/`.

Synccheck and Racecheck keep their complete-kernel tests under each tool's
`corpus/` directory.
