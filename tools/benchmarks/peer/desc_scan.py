"""Scan compiled kernels for global accesses through a half-written memory
descriptor.

On sm_100 a global load or store names a 64-bit memory descriptor held in a
uniform register pair, ``desc[URn]``. NVRTC 13.0 can leave one half of the
pair unwritten (docs/numsim/MULTI_GPU_PEER.md); the access then uses whatever
an earlier kernel left in that register. This flags every access whose pair
has a half that no earlier instruction of the function writes. "Earlier" is
program order, not control flow, so a clean scan is evidence, not proof.

    python -m benchmarks.peer.desc_scan FILE...        # .so, .cubin or .fatbin
    python -m benchmarks.peer.desc_scan build all_gather [--mode nvrtc] [--shard 0/8]

``build`` compiles, on every rank, every TIRx configuration a benchmark times
(the all-gather tuning log's correct ones, or the committed results of the
multimem GEMM + all-reduce and two-shot all-reduce) the way that benchmark
builds it, and scans each library. It reads the SM count from GPU 0.
"""

from __future__ import annotations

import argparse
import json
import re
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FATBIN_MAGIC = b"\x50\xed\x55\xba"
EM_CUDA = 190
SHT_NOBITS = 8
DESC = re.compile(r"(?<![gi])desc\[UR(\d+)\]")
INSN = re.compile(r"/\*([0-9a-f]{4,})\*/\s+(.*?)\s*;")
# The benchmarks' default compile modes: the all-gather port builds with nvcc.
MODES = {"all_gather": "nvcc", "gemm_all_reduce": "nvrtc", "two_shot": "nvrtc"}


def _device_images(data: bytes) -> list[tuple[str, bytes]]:
    """The fatbins and CUDA ELF images embedded in a host library."""

    images, fatbins = [], []
    start = 0
    while (p := data.find(FATBIN_MAGIC, start)) >= 0:
        _, _, header, size = struct.unpack_from("<IHHQ", data, p)
        images.append((".fatbin", data[p:p + header + size]))
        fatbins.append(range(p, p + header + size))
        start = p + header + size
    start = 1
    while (p := data.find(b"\x7fELF\x02\x01", start)) >= 0:
        start = p + 1
        if struct.unpack_from("<H", data, p + 0x12)[0] != EM_CUDA or any(p in f for f in fatbins):
            continue
        phoff, shoff = struct.unpack_from("<QQ", data, p + 0x20)
        phentsize, phnum, shentsize, shnum = struct.unpack_from("<HHHH", data, p + 0x36)
        end = max(phoff + phentsize * phnum, shoff + shentsize * shnum)
        for i in range(shnum):
            section = p + shoff + i * shentsize
            if struct.unpack_from("<I", data, section + 4)[0] != SHT_NOBITS:
                offset, size = struct.unpack_from("<QQ", data, section + 0x18)
                end = max(end, offset + size)
        images.append((".cubin", data[p:p + end]))
        start = p + end
    return images


def sass(path: Path) -> str:
    """The SASS of every device image in ``path``."""

    data = path.read_bytes()
    images = [(path.suffix, data)] if path.suffix in (".cubin", ".fatbin") else _device_images(data)
    texts = []
    for suffix, image in images:
        with tempfile.NamedTemporaryFile(suffix=suffix) as f:
            f.write(image)
            f.flush()
            texts.append(subprocess.run(["cuobjdump", "-sass", f.name], capture_output=True,
                                        text=True, check=True).stdout)
    return "\n".join(texts)


def _writes(insn: str, reg: int) -> bool:
    """Whether ``insn`` writes uniform register ``reg``, as its destination or
    the high half of a 64-bit destination."""

    parts = re.sub(r"^@!?U?P\w+\s+", "", insn).split(None, 1)
    if len(parts) < 2:
        return False
    op, operands = parts
    m = re.fullmatch(r"UR(\d+)", operands.split(",")[0].strip())
    if not m:
        return False
    first = int(m.group(1))
    wide = ".64" in op or op.startswith("UIMAD.WIDE")
    return reg == first or (wide and reg == first + 1)


def scan(text: str) -> tuple[int, list[str]]:
    """``(functions, findings)``: one finding per access through a register pair
    with an unwritten half."""

    chunks = re.split(r"\n\s*Function : ", text)[1:]
    findings = []
    for chunk in chunks:
        name = chunk.split("\n", 1)[0].strip()
        insns = INSN.findall(chunk)
        for i, (offset, insn) in enumerate(insns):
            for reg in map(int, DESC.findall(insn)):
                unwritten = [half for half in (reg, reg + 1)
                             if not any(_writes(insns[j][1], half) for j in range(i))]
                if unwritten:
                    findings.append(f"{name} {offset}: {insn} (UR{unwritten[0]} never written)")
    return len(chunks), findings


def _scan_library(executable) -> tuple[int, list[str]]:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "kernel.so"
        executable.export_library(str(path))
        return scan(sass(path))


def _all_gather_builds(world: int, clusters: dict[int, int]):
    from benchmarks.peer.all_gather_gemm import RESULTS, SHAPES_BY_NAME
    from ported.cutlass.sm100_all_gather_gemm import all_gather_gemm, barrier_kernel

    seen = set()
    for line in (RESULTS / "all_gather_gemm_tuning.jsonl").read_text().splitlines():
        row = json.loads(line)
        if row["impl"] != "tirx" or row["status"] != "ok" or (row["shape"], row["config"]) in seen:
            continue
        seen.add((row["shape"], row["config"]))
        shape, config = SHAPES_BY_NAME[row["shape"]], json.loads(row["config"])
        cluster = tuple(config["cluster"])
        chunk_rows = config.get("chunk_rows", shape["m"])
        for rank in range(world):
            yield f"r{rank} {row['shape']} {row['config']}", lambda rank=rank, shape=shape, \
                config=config, cluster=cluster, chunk_rows=chunk_rows: [
                    all_gather_gemm(
                        shape["m"], shape["n"], shape["k"], shape["ab"], shape["c"], rank, world,
                        use_2cta=config["use_2cta"], mma_tiler=tuple(config["mma_tiler"]),
                        cluster=cluster, use_tma_store=config["use_tma_store"],
                        raster=config.get("raster", "m"), swizzle=config.get("swizzle", 1),
                        chunk_rows=chunk_rows,
                        max_active_clusters=clusters[cluster[0] * cluster[1]]).func,
                    barrier_kernel(rank, world, world * (shape["m"] // chunk_rows)).func]


def _gemm_all_reduce_builds(world: int, clusters: dict[int, int]):
    from benchmarks.multimem.gemm_all_reduce import RESULTS, SHAPES_BY_NAME
    from tests.numsim.support.multimem_gemm_all_reduce import gemm_all_reduce

    seen = set()
    for line in (RESULTS / "gemm_all_reduce_gb200x4.jsonl").read_text().splitlines():
        row = json.loads(line)
        key = (row["shape"], row["origin"], row["config"])
        if key in seen:
            continue
        seen.add(key)
        shape, config = SHAPES_BY_NAME[row["shape"]], json.loads(row["config"])
        cluster = tuple(config["cluster"])
        for rank in range(world):
            yield f"r{rank} " + " ".join(key), lambda rank=rank, shape=shape, config=config, \
                cluster=cluster, protocol=row["origin"]: [gemm_all_reduce(
                    shape["m"], shape["n"], shape["k"], shape["ab"], shape["c"], rank, world,
                    use_2cta=config["use_2cta"], mma_tiler=tuple(config["mma_tiler"]),
                    cluster=cluster, use_tma_store=config["use_tma_store"],
                    raster=config.get("raster", "m"), swizzle=config.get("swizzle", 1),
                    protocol=protocol, max_active_clusters=clusters[cluster[0] * cluster[1]],
                    flag_len=(shape["m"] // 64) * (shape["n"] // 64) + 160).func]


def _two_shot_builds(world: int, clusters: dict[int, int]):
    from benchmarks.multimem.two_shot_all_reduce import SHAPES
    from tests.numsim.support.multimem_kernels import TILE, two_shot_all_reduce

    for shape in SHAPES:
        yield f"{shape[0]}x{shape[1]}", lambda shape=shape: [
            two_shot_all_reduce(shape[0] // TILE, world, shape[1] // TILE)]


BUILDS = {"all_gather": _all_gather_builds, "gemm_all_reduce": _gemm_all_reduce_builds,
          "two_shot": _two_shot_builds}


def build(which: str, mode: str, world: int, shard: str) -> int:
    import torch
    from tirx_kernels.runner import compile_kernel

    from benchmarks.multimem.origins import max_active_clusters

    torch.zeros(1, device="cuda")
    clusters = {size: max_active_clusters(size) for size in (1, 2, 4, 8)}
    index, count = (int(x) for x in shard.split("/"))
    jobs = list(BUILDS[which](world, clusters))[index::count]
    suspect = 0
    for label, funcs in jobs:
        findings = []
        for func in funcs():
            functions, found = _scan_library(compile_kernel(func, arch="sm_100a",
                                                            cuda_compile_mode=mode))
            if not functions:
                raise RuntimeError(f"{label}: no device code in the library")
            findings += found
        suspect += bool(findings)
        print(("SUSPECT " if findings else "ok      ") + label, *findings[:2], flush=True)
    print(f"{which} ({mode}, shard {shard}): {suspect} of {len(jobs)} builds suspect", flush=True)
    return suspect


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "build":
        parser = argparse.ArgumentParser(description="scan every benchmarked TIRx build")
        parser.add_argument("build")
        parser.add_argument("which", choices=sorted(BUILDS))
        parser.add_argument("--mode", choices=["nvrtc", "nvcc"])
        parser.add_argument("--world", type=int, default=4)
        parser.add_argument("--shard", default="0/1")
        args = parser.parse_args()
        suspect = build(args.which, args.mode or MODES[args.which], args.world, args.shard)
        sys.exit(1 if suspect else 0)
    suspect = 0
    for path in map(Path, sys.argv[1:]):
        functions, findings = scan(sass(path))
        print(f"{path}: {functions} functions, {len(findings)} suspect accesses")
        for finding in findings:
            print("  " + finding)
        suspect += bool(findings) or not functions
    sys.exit(1 if suspect else 0)


if __name__ == "__main__":
    main()
