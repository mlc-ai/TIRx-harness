---
name: tirx-wiki
description: >
  Authoritative reference for TIRx-lite authoring on TIRx: TIRx-lite APIs, canonical
  kernels, optimization guidance, PTX ISA, and GPU references. Highlight the
  TIRx-lite guide, canonical kernels, and PTX manual when relevant, while retaining
  the other reference sources. Pair it with tirx-debug-kernel or
  tirx-profile-kernel for runnable diagnosis or measurement.
---

# TIRx Wiki

## Route operational work

Keep this skill focused on authoritative knowledge and source lookup.

- Invoke `$tirx-debug-kernel` for wrong results, hangs, synchronization,
  races, numerical tracing, or sanitizer work on a runnable kernel.
- Invoke `$tirx-profile-kernel` for benchmarks, IKET, NCU, or generated
  CUDA/PTX/SASS inspection.
- Invoke both operational skills when a performance change also touches
  synchronization or cross-thread-visible memory.

Loading a tool page or quoting its example is not evidence that the tool ran.
The operational skills own execution, artifact collection, reruns, completion
gates, and their detailed tool guides.

## Use the wiki

1. Read `references/repos/tirx-kernels/README.md` for the kernel index and
   browse the source and documentation in that checkout. If it is missing,
   run the reference fetcher below.
2. Use the TIRx-lite guide for the authoring contract, the kernel checkout
   for implementation patterns, and the PTX ISA manual for instruction
   semantics when those are relevant.
3. Use the optimization manual, NVIDIA manuals, blogs, and live reference
   repositories for the corresponding optimization, hardware, or
   external-source question.
4. Verify API claims against installed TIRx-lite/TIRx and a canonical call site;
   mark uncertain hardware claims `[VERIFY]`.
5. Search with `rg` when the relevant page is not obvious.

In the optimization references, resolve `tirx-kernels/...` under
`references/repos/`. The checkout follows the upstream default branch. Use
the pip-installed package to verify runtime APIs when the revisions differ.

## Materialize live references

If a needed manual or external repository is absent, run from this skill
directory:

```bash
python -m pip install -r requirements.txt
python scripts/fetch_references.py
```

Rerun the fetcher to replace the materialized manuals and repository checkouts
with the versions currently served by their declared upstream URLs. Eval setup
invokes this same script while excluding resources banned by the task.

## Highlighted references

- Kernel index: `references/repos/tirx-kernels/README.md`
- TIRx-lite authoring guide: `references/repos/tirx-kernels/tirx_kernels/tirx_lite/README.md`
- Kernel implementations: `references/repos/tirx-kernels/tirx_kernels/`
- PTX ISA manual: `references/manuals/ptx_isa.rst`

## Additional references

- [Optimization manual](optimization/INDEX.md)
- [Reference corpus index](references/INDEX.md)
- [NVIDIA manuals](references/manuals/INDEX.md)
- [GPU engineering blogs](references/blogs/INDEX.md)
- [Live external repositories](references/repos/INDEX.md)

## Package boundary

Use the fetched checkout for reference reading and the pip-installed
`tirx_kernels` package for execution. Runnable debug and profiling procedures
belong to their owning skills.
