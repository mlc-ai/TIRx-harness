"""The 39 user-specified prepared model-forward workloads."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Case:
    name: str
    group: str
    segments: tuple[tuple[int, int, int], ...]  # (batch, previous, new)

    @property
    def previous(self):
        return [previous for batch, previous, _ in self.segments for _ in range(batch)]

    @property
    def lengths(self):
        return [new for batch, _, new in self.segments for _ in range(batch)]

    @property
    def batch(self):
        return sum(batch for batch, _, _ in self.segments)

    @property
    def all_logits(self):
        return self.name in ("D12", "D13")

    @property
    def graph(self):
        return self.group == "decode"


PREFILL = [(1,0,4096), (1,0,8192), (1,0,32768), (1,8192,8192),
           (1,32768,8192), (1,65536,8192), (1,114688,8192), (1,32768,32768),
           (1,90112,32768), (4,0,8192), (8,0,4096), (2,32768,16384), (4,16384,8192)]
DECODE = [(1,2048,1), (8,4096,1), (64,4096,1), (128,4096,1), (256,2048,1),
          (128,8192,1), (64,16384,1), (32,32768,1), (16,65536,1), (8,122880,1),
          (48,32768,1), (64,4096,4), (16,65536,4)]
EXPAND = [(1,0,256), (16,0,512), (64,0,512), (256,0,128), (16,8192,256),
          (64,8192,256), (128,4096,128), (32,32768,512), (8,98304,512), (1,98304,2048)]
CASES = [Case(f"{prefix}{i}", group, (shape,))
         for prefix, group, shapes in (("P","prefill",PREFILL), ("D","decode",DECODE), ("S","expand",EXPAND))
         for i, shape in enumerate(shapes, 1)]
CASES += [Case("S11", "expand", ((16,8192,256), (64,8192,1))),
          Case("S12", "expand", ((1,98304,2048), (128,4096,1))),
          Case("S13", "expand", ((4,0,512), (128,4096,1)))]


def select(group="all", names=None):
    if names:
        unknown = set(names) - {c.name for c in CASES}
        if unknown:
            raise ValueError(f"Unknown cases: {sorted(unknown)}")
    return [c for c in CASES if (group == "all" or c.group == group) and (not names or c.name in names)]
