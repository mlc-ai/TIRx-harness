# Simulation inputs

These types describe NumSim inputs and numerical comparisons. Unless otherwise
noted, import them from `tirx_harness.numsim`. Fields and defaults below are
read from the Python source. Construct each type with the arguments
documented below.

## Multi-GPU bindings

For a single-node multi-rank launch, pass a list of per-rank input dictionaries
to `Engine.run`, `racecheck`, or `synccheck`. Every rank uses the same kernel
and grid. Use `MulticastWindow(replicas)` for a multicast address and
`SymmetricBuffer(replicas)` for peer-accessible memory. Each takes one
C-contiguous NumPy array of matching dtype and shape per rank.

`SymmetricBuffer.peer_offsets(rank)` returns the byte offsets from that rank's
replica to its peers. Results use `rank_binding_name(name, rank)`, for example
`"out@rank2"`. A multicast window itself has no output; bind its unicast
replica to inspect the result.

The {repo}`multimem guide <tools/docs/numsim/MULTI_GPU_MULTIMEM.md>` and
{repo}`peer-memory guide <tools/docs/numsim/MULTI_GPU_PEER.md>` document the
binding restrictions, memory-ordering rules, and related tests.

## Reusable cases and comparisons

```{eval-rst}
.. autoapiclass:: tirx_harness.numsim.NumSimCase
   :members:
   :undoc-members:
```

`kernel` is the specialized TIRx function; `args` binds its parameters,
including output storage. `outputs` selects the result buffers. `reference`
is a zero-argument callable returning a dictionary of expected outputs;
`comparisons` maps output names to comparison specifications. Pass the case to
{py:func}`~tirx_harness.numsim.run_case`.

```{eval-rst}
.. autoapiclass:: tirx_harness.numsim.ComparisonSpec
   :members:
   :undoc-members:
```

`rtol` and `atol` are finite, nonnegative relative and absolute tolerances.
`equal_nan` controls whether matching not-a-number values compare equal.
`actual_encoding="bfloat16"` decodes a simulated output's integer backing
storage before comparison. `regions` optionally restricts the compared areas.

```{eval-rst}
.. autoapiclass:: tirx_harness.numsim.ComparisonRegion
   :members:
   :undoc-members:
```

`actual` and `expected` are tuples of integer indices or slices. If `expected`
is omitted, the same selection is used for both arrays. Selected regions must
be nonempty and have matching shapes.

## Tensor maps

A tensor map describes the storage and tile shape for a tensor-memory transfer.
Use `TensorMap(...).numpy()` to create a simulator descriptor, then bind that
array under the kernel's tensor-map parameter name. This descriptor is for
CPU simulation.

```{eval-rst}
.. autoapiclass:: tirx_harness.numsim.TensorMap
   :members:
   :undoc-members:
```

Shapes and element strides follow the descriptor's dimension order, with the
innermost dimension first; global strides are in bytes. Tensor Memory
Accelerator (TMA) dtype overrides include TensorFloat-32 (`tf32`),
flush-to-zero (`ftz`) floating-point modes, and packed six-bit unsigned values
(`uint6`). `fp4_shared_layout` selects a four-bit floating-point storage layout.

```{eval-rst}
.. autoapiclass:: tirx_harness.numsim.Im2col
   :members:
   :undoc-members:
```

`Im2col` describes an image-to-column transfer for convolution. Its spatial
coordinates use width, height, and depth order (W/H/D).

## Launch selection

Most callers execute the full launch. To select complete clusters or thread
blocks, import `ExecutionSubset` from `tirx_harness.numsim.api` and pass it as
`subset` to `Engine.run`. A cooperative thread array (CTA) is a thread block.
For a multi-kernel launch, use a mapping from phase indices to selections.

```{eval-rst}
.. autoapiclass:: tirx_harness.numsim.api.ExecutionSubset
   :members: cluster_ids, cta_ids
   :undoc-members:
```

```{eval-rst}
.. autoapidata:: tirx_harness.numsim.api.ExecutionSubsetSelection
```

This type alias accepts an `ExecutionSubset` or a mapping from integer phase
indices to `ExecutionSubset` objects.

```{eval-rst}
.. autoapiclass:: tirx_harness.numsim.ExecutionAssumptions
   :members: external_grid_dependencies_satisfied
   :undoc-members:
```
