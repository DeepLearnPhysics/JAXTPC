# Padding & masking

JAXTPC runs the same compiled JIT function for every event. That only works if
every array has the **same shape** on every call — but real events have wildly
different deposit counts. The fix is fixed-size padding plus a single masking
point that neutralizes the padding. Understanding this is essential: the
padding contract is what lets one compilation serve an entire dataset, and the
divisibility rule below is a footgun that bites at construction time.

## Fixed shapes: `total_pad` per volume

Each volume's deposit arrays are padded to a fixed length, `total_pad`, at load
time (`tools/loader.py`). Padding to a constant means the shapes JAX sees never
change, so the simulator is traced and compiled **once** and then reused for
every subsequent event — no recompilation per event.

`total_pad` is a capacity: if a volume has more real deposits than `total_pad`,
the loader raises immediately rather than silently truncating (see
[capacities](capacities.md)). Real deposits occupy the first `n_actual` slots;
the remaining `total_pad - n_actual` slots are padding.

## Padding entries carry sentinel values

When `_build_padded_deposit_data` pads a volume (`tools/loader.py`), padding
slots are filled with values chosen so they cannot produce signal:

| Field | Padding value | Why |
|---|---|---|
| `de` (energy deposited) | `0` | zero energy → zero recombined charge |
| `dx` (step length) | `1` | avoids divide-by-zero in `dE/dx` (never `0`) |
| `track_ids` | `-1` | invalid track sentinel |
| `interaction_ids`, `root_track_ids` | `-1` | invalid sentinels |

`n_actual` is stored on the `VolumeDeposits` so downstream code knows how many
leading entries are real.

## The single masking point

There is exactly **one** place where padding is masked out: in
`compute_volume_physics` (`tools/physics.py`), right after recombination. Every
deposit — real and padding — goes through recombination, then the result is
multiplied by a boolean mask that is `True` only for indices below `n_actual`:

```python
# Zero out padding entries. n_actual is the count of real deposits;
# everything beyond that is padding and must not contribute signal.
# This is the single masking point — all downstream code trusts charges=0 for padding.
padding_mask = jnp.arange(deposits.de.shape[0]) < deposits.n_actual
charges = charges * padding_mask
photons = photons * padding_mask
```

After this line, `charges` (and `photons`) are exactly `0` for every padding
slot.

!!! note "Downstream trusts `charges = 0` — no re-masking"
    Nothing after this point re-checks `n_actual`. Drift, per-plane
    correction, wire projection, response, and accumulation all run over the
    full padded array, but a padding deposit carries `charge = 0`, so its
    response contribution (`intensity = charge × attenuation`) is zero and it
    scatter-adds nothing. `compute_chunk_response` even passes
    `valid_hit=True` for every deposit on purpose — the charge, not a per-hit
    flag, is what does the masking. The bucketed and box paths rely on the same
    invariant: `compute_bucket_maps` notes it needs "no valid_mask… padding
    entries have charges=0."

    A `de = 0` sentinel that is *not* masked would still be safe (zero energy
    → zero charge), but the explicit mask makes the guarantee unconditional and
    independent of the recombination model.

The same in-window pattern reappears in `compute_plane_physics`: charges for
deposits whose readout tick falls outside the window are zeroed
(`charges = vol_int.charges * in_window`), reusing the "zero the charge, trust
it downstream" idiom rather than threading another mask through the pipeline.

## Chunk divisibility

Accumulation runs as a `jax.lax.fori_loop` over fixed-size chunks of the padded
array (`compute_plane_signal` in `tools/physics.py`). The loop takes
`total_pad // chunk_size` steps, so the chunk size must divide `total_pad`
evenly — otherwise the tail of the array would be dropped or over-run. The
simulator enforces this at construction (`tools/simulation.py`):

- `total_pad` must be divisible by `response_chunk_size` (sensor response loop).
- `total_pad` must be divisible by `hits_chunk_size` (track-hits loop, when
  `include_track_hits=True`).

!!! warning "Divisibility footgun"
    A mismatched chunk size raises a `ValueError` **at simulator
    construction**, before any event runs:

    ```text
    total_pad (450,000) must be divisible by response_chunk_size (50,000).
    total_pad (450,000) must be divisible by hits_chunk_size (25,000).
    ```

    Pick `total_pad` and the chunk sizes together so the division is exact
    (e.g. `total_pad = 450_000`, `response_chunk = hits_chunk = 28_125`). The
    [profiler](../production/profiler.md) always emits a consistent set, so
    this only bites when hand-editing a production config.

## Why this design

Folding all masking into one multiply keeps the pipeline branch-free and
uniform: the JIT body has no data-dependent control flow over "is this deposit
real?", every volume and plane runs the identical traced graph, and the only
per-event quantity is `n_actual` (which bounds the `fori_loop` trip count via
`jnp.minimum`, so padding chunks aren't even executed). See
[coordinates](coordinates.md) for the companion invariant (one canonical
frame) and [capacities](capacities.md) for how `total_pad` and the chunk sizes
are sized and what happens on overflow.
