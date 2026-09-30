"""Exercise an installed wheel without importing checkout sources or using a GPU."""

import subprocess
import sys
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

import tirx_harness
from tirx_harness import numsim
from tirx_harness.numsim.transpiler import native_frontend
from tvm.script import tirx as T


@T.prim_func
def add_one(source: T.Buffer((32,), "float32"), output: T.Buffer((32,), "float32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    output[lane] = source[lane] + T.float32(1)


def main():
    package = Path(tirx_harness.__file__).resolve().parent
    assert "site-packages" in package.parts, package
    assert tirx_harness.__version__ == version("tirx-harness")
    assert callable(tirx_harness.racecheck) and callable(tirx_harness.synccheck)
    library = native_frontend.library_path()
    assert library.is_relative_to(package), library
    assert native_frontend._library() is not None
    for name in ("LICENSE", "NOTICE"):
        assert (library.parent / "_thirdparty_licenses" / "tvm-rust-ext" / name).is_file()

    source = np.arange(32, dtype=np.float32)
    with TemporaryDirectory() as temporary:
        module = numsim.transpile(add_one, cache_dir=Path(temporary))
        result = numsim.Engine(max_workers=1).run(
            module, {"source": source, "output": np.zeros_like(source)}
        )
        np.testing.assert_array_equal(result.outputs["output"], source + 1)

        skills_dir = Path(temporary) / "skills"
        # The console script sits beside the interpreter of the test venv.
        subprocess.run(
            [
                str(Path(sys.executable).with_name("tirx-harness")),
                "skills",
                "install",
                "--dest",
                str(skills_dir),
                "--no-fetch",
            ],
            check=True,
        )
        for name in ("tirx-debug-kernel", "tirx-profile-kernel", "tirx-wiki"):
            assert (skills_dir / name / "SKILL.md").is_file(), name
    print(
        f"tirx-harness {version('tirx-harness')}: installed native frontend, NumSim, and skills OK"
    )


if __name__ == "__main__":
    main()
