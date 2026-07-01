# FAQ & common errors

A lookup table for the errors people actually hit, with the real (or closely
paraphrased) message text so you can match on it. Capacity sizing is the
[profiler](../production/profiler.md)'s job and the failure modes are explained in
[capacities](../concepts/capacities.md) — those two pages are canonical; this page
just maps a symptom to a fix.

## Quick index

| Symptom / message contains… | Cause | Fix |
|---|---|---|
| `deposits > total_pad` at load | too many deposits in a volume | raise `total_pad` → [capacities](../concepts/capacities.md#total_pad-deposits-per-volume) |
| `maxg overflow vol N` | too many groups this event | raise `maxg` → [capacities](../concepts/capacities.md#maxg-maxg_medium-groups-per-event) |
| `track_hits overflow … max_keys` | too many non-zero box cells | raise `max_keys` → [capacities](../concepts/capacities.md#max_keys-track-hits-box-cell-budget) |
| `Bucket overflow … max_active_buckets` | too many active buckets (bucketed mode) | raise `max_buckets` → [capacities](../concepts/capacities.md#max_buckets-active-buckets-bucketed-mode) |
| `total_pad … must be divisible by …_chunk_size` | chunk doesn't divide `total_pad` | fix chunk sizes → [capacities](../concepts/capacities.md#response_chunk-hits_chunk-chunk-sizes) |
| `Unable to synchronously open … filter` / can't read HDF5 | missing `hdf5plugin` import | `import hdf5plugin` before `h5py` |
| notebook renders but shows no plots | outputs stripped by `nbstripout` | run the notebook (`Run All`) |
| sim runs on CPU / too slow / OOM on GPU | JAX device selection | set `JAX_PLATFORM_NAME` |

## Capacity errors

All capacity overflows point you at `profiler.setup_production`, which sizes the
whole set for your dataset. Run it once and pass the result with
`--production-config`:

```bash
python3 -m profiler.setup_production --data <run_dir/> --config <detector.yaml> -o <production.yaml>
```

### `total_pad` exceeded at load

```text
Volume 0 has 512,340 deposits > total_pad (450,000).
Increase --total-pad or run profiler.setup_production.
```

Raised in `tools/loader.py` **before the event reaches the GPU** — hard crash.
Size `total_pad` to the max deposit count over the dataset. See
[capacities → total_pad](../concepts/capacities.md#total_pad-deposits-per-volume).

### Too many groups → raise `maxg`

```text
maxg overflow vol 0: n_groups=118,204 >= maxg=110,000.
Increase --maxg or run profiler.setup_production.
```

Raised on the host in `process_event`. In batch, `run_batch` **catches this, logs
it as `maxg_overflow`, skips the event**, and reprocesses at a higher `maxg` — so
a rare tail event is not fatal. Bump `maxg` (or the tiered `maxg_medium`). See
[capacities → maxg](../concepts/capacities.md#maxg-maxg_medium-groups-per-event).

### `max_keys` overflow → raise `max_keys`

```text
track_hits overflow vol 0 plane 2: count=9,412,880 >= max_keys=9,000,000.
Increase --max-keys or run profiler.setup_production.
```

Hard crash in `process_event`. The reported `count` is the **true** cell count, so
set `max_keys` above it. `max_keys` is charge-dependent (the profiler estimates it
without simulating; see the calibration knobs). See
[capacities → max_keys](../concepts/capacities.md#max_keys-track-hits-box-cell-budget).

### Too many active buckets → raise `max_buckets`

```text
Bucket overflow vol 0 plane 2: num_active=1,024 >= max_active_buckets=1,000.
Increase --max-buckets.
```

Only happens in **bucketed** (wire) accumulation mode (`--bucketed`). Inert for
pixel. See [capacities → max_buckets](../concepts/capacities.md#max_buckets-active-buckets-bucketed-mode).

### `total_pad` not divisible by a chunk size

```text
total_pad (450,000) must be divisible by response_chunk_size (50,000).
total_pad (450,000) must be divisible by hits_chunk_size (25,000).
```

A `ValueError` at **simulator construction** (before any event runs): both
`response_chunk_size` and `hits_chunk_size` must divide `total_pad` evenly. Pick
chunk sizes that are factors of `total_pad` (the profiler keeps them divisible).
See [capacities → chunk sizes](../concepts/capacities.md#response_chunk-hits_chunk-chunk-sizes).

## Reading production output fails

Production HDF5 is written with the `blosc-zstd` codec by default (smaller and
faster than gzip). Reading it requires the `hdf5plugin` filters to be registered:

```text
OSError: Unable to synchronously open object (can't read data: filter returned failure during read)
```

!!! tip "Fix: import `hdf5plugin` first"
    ```python
    import hdf5plugin  # registers blosc/zstd/lz4 filters — must come before h5py reads
    import h5py
    ```
    `production/load.py` (and pimm-data's readers) register it automatically; an
    ad-hoc `h5py` consumer must import it itself. Alternatively write with
    `--codec gzip`, which needs no plugin but is larger and slower.

## Notebooks show no plots

The repo uses an `nbstripout` filter (`.gitattributes: *.ipynb filter=nbstripout`)
that strips **all cell outputs** on commit, so notebooks in a fresh checkout have
no rendered figures. This is intentional (keeps diffs clean). To see the plots,
open the notebook and run it (`Run All`) — every notebook is CI-runnable on
synthetic data, so no external files are needed.

## CPU vs GPU (JAX device selection)

JAXTPC runs on whichever backend JAX picks. Force it with the `JAX_PLATFORM_NAME`
environment variable:

```bash
# Tests, quick synthetic runs, or a machine with no GPU:
JAX_PLATFORM_NAME=cpu python3 -m pytest tests/ -v -m "not slow"

# Default (GPU) production run:
python3 production/run_batch.py --data events.h5 --events 100
```

- **Running on CPU unexpectedly / very slow?** JAX fell back to CPU — check your
  CUDA-enabled `jax`/`jaxlib` wheel is installed and a GPU is visible.
- **GPU out-of-memory?** Capacities scale sim memory; size them with the
  [profiler](../production/profiler.md) rather than over-provisioning, and note
  save workers are GPU-memory bound (see [batch](../production/batch.md)).
- The full test suite is **CPU-only on synthetic data** by design — always set
  `JAX_PLATFORM_NAME=cpu` for tests.
</content>
