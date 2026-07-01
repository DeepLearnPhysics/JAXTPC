# API reference

Auto-generated from the source docstrings (numpy style), grouped by concept.
For the *why* behind these APIs, start with the [Reading guide](../../architecture/reading-guide.md).

- [Core simulation](core.md) — `DetectorSimulator` and its entry points
- [Data types](data-types.md) — the `NamedTuple` parameter/state bundles
- [Loading & config](loading.md) — YAML → config → `DepositData`
- [Physics](physics.md) — recombination, drift, the shared body
- [Response & readout](response.md) — kernels, wires, output formats
- [Post-processing](post-processing.md) — electronics, noise, SCE
- [Truth](truth.md) — track-hit correspondence
- [Differentiable](differentiable.md) — losses, particle generation
- [Visualization](visualization.md) — plotting helpers
- [Production I/O](production.md) — HDF5 writers/readers

!!! note
    A few functions are kept as documented **reference / alternate-mode** code
    (e.g. the legacy standalone track-hits path, the bucketed accumulators).
    Their docstrings say so — they are not the default production path.
