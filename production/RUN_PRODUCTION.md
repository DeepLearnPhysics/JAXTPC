# Running Production

How to drive `run_batch.py` over a full dataset, including multi-shard
distribution and overflow handling. Read this before launching long runs.

## Quick start (single GPU, all files)

```bash
python3 production/run_batch.py \
    --data /sdf/data/neutrino/doraemon/test_00_00_01/run_*/ \
    --config config/cubic_wireplane_config.yaml \
    --production-config config/production_cubic_wireplane_doraemon.yaml \
    --dataset sim \
    --outdir /sdf/home/o/omara/neutrino_data/omara/doraemon \
    --edepsim-ids
```

`--data` accepts files, directories (globbed for `*.h5`), or glob patterns.
Multiple paths can be given; they're concatenated.

## Distributing across multiple GPUs / SLURM jobs

Use `--shard-id N --num-shards K` to give each job 1/K of the files
(round-robin). The shards are independent — they write to different
per-run subfolders or, in rare overlap cases, different output files.

```bash
# Launch 4 shards across 4 GPUs (one job per GPU)
for S in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=$S python3 production/run_batch.py \
        --data /sdf/data/neutrino/doraemon/test_00_00_01/run_*/ \
        --config config/cubic_wireplane_config.yaml \
        --production-config config/production_cubic_wireplane_doraemon.yaml \
        --outdir /sdf/home/o/omara/neutrino_data/omara/doraemon \
        --shard-id $S --num-shards 4 \
        --edepsim-ids &
done
wait
```

Or use `--file-range "0:100"` (Python slice) for an explicit range. Range
is applied AFTER shard selection, so `--shard-id 0 --num-shards 4 --file-range 0:10`
processes the first 10 files of shard 0.

## Output structure

```
<outdir>/
├── sensor/run_<run_id>/sim_sensor_<NNNN>.h5      # sparse digitized readout
├── step/run_<run_id>/sim_step_<NNNN>.h5          # truth deposits
├── hits/run_<run_id>/sim_hits_<NNNN>.h5          # per-particle correspondence
└── logs/
    ├── overflow_events_shard<SSS>.csv            # one row per skipped event
    └── summary_shard<SSS>.txt                    # final tally per shard
```

`<NNNN>` matches the source `edepsim_<NNNN>.h5` file index for direct
traceability back to the input. `<run_id>` is the 10-digit zero-padded
run identifier parsed from the source path (e.g. `run_0026628546`).

If `events-per-file` is smaller than the source's event count and a
single source produces multiple output files, the suffix becomes
`<NNNN>_<II>.h5`.

## Output compression

All datasets are compressed with the codec set by `--codec` (default
**`blosc-zstd`**, level 4 + byte shuffle). The loader profiler found gzip is
Pareto-dominated: blosc-zstd is ~6–20% smaller than gzip **and** ~2.3× faster
to read and write. Options:

| `--codec` | vs gzip |
|---|---|
| `blosc-zstd` (default) | smaller + ~2.3× faster read/write |
| `blosc-lz4hc` | gzip's size, ~4× faster read (slower write) |
| `blosc-lz4` | fastest read+write, ~19% larger |
| `gzip` / `gzip-1` / `lzf` | legacy / no plugin needed |

**Reading non-gzip output requires `import hdf5plugin`.** It's a declared
dependency of pimm-data and is auto-registered by `production/load.py` and
pimm-data's readers, so the normal load paths "just work". Ad-hoc `h5py`
consumers must import it themselves; a missing `hdf5plugin` makes the writer
**raise** rather than silently fall back to gzip. (The browser viewer's
h5wasm build does not yet support blosc — see `viewer/`.) Re-encode existing
files between codecs with `pimm-data/scripts/transcode_codec.py`.

## Overflow handling

`max_keys` and `total_pad` can be exceeded by events larger than the
production config was tuned for. The original code would crash on the
first such event; this version:

1. **Catches `RuntimeError`** from `load_deposit` and `process_event`.
2. **Classifies** the error: `total_pad_overflow`, `max_keys_overflow`,
   `max_buckets_overflow`, or `runtime_error`.
3. **Appends one row** to `overflow_events_shard<SSS>.csv` with:
   `timestamp, source_path, src_file_idx, event_idx, event_id,
   n_deposits, error_type, error_message`.
4. **Continues** with the next event. The output HDF5 files do NOT
   contain entries for skipped events.

To audit a run:

```bash
# How many were skipped, by type?
wc -l <outdir>/logs/overflow_events_shard*.csv
awk -F, 'NR>1 {print $7}' <outdir>/logs/overflow_events_*.csv | sort | uniq -c

# Worst offenders by deposit count:
awk -F, 'NR>1 {print $6, $0}' <outdir>/logs/overflow_events_*.csv | sort -rn | head
```

If you see many overflows, re-run the profiler with `--use-max` and a
larger `--probe-events`, then re-tune (`max_keys`, `total_pad`).

## Smoke test before committing to a long run

Always run on 1–2 source files first to catch pipeline bugs:

```bash
python3 production/run_batch.py \
    --data /sdf/data/.../run_0026628546/edepsim_000000.h5 \
    --config config/cubic_wireplane_config.yaml \
    --production-config config/production_cubic_wireplane_doraemon.yaml \
    --outdir /tmp/jaxtpc_smoke \
    --events 5
```

Verify: the three HDF5 files exist, no entries in
`logs/overflow_events_shard000.csv`, `summary_shard000.txt` reports
`events_processed=5 events_skipped=0`.

## Production configs

Production parameters are tuned per detector. Generated by
`profiler.setup_production` and stored as YAML in `config/`:

- `config/production_cubic_wireplane_doraemon.yaml` — doraemon **wire**
  dataset (large events, ~1M deposits typical; total_pad=1.33M, max_keys=7M).
- `config/production_cubic_pixel_doraemon.yaml` — doraemon **pixel** readout
  (total_pad=1.33M, hits_chunk=66.5k, max_keys=8M). ~0.4% of the densest
  events still exceed max_keys and are logged + skipped (true max ~14M keys).
- `config/production_cubic_wireplane_config.yaml` — generic wire setup for
  smaller events (total_pad=970k, max_keys=6M).

Each `hits_chunk` sits near the optimum of the 2D timing landscape for that
`total_pad`. `--production-config` loads `total_pad`, `response_chunk`,
`hits_chunk`, `max_keys`, `max_buckets`, and the thresholds into argparse
defaults; CLI flags override the file values if you want to experiment.
(The output codec is **not** carried by the production config — set it with
`--codec`; see *Output compression* below.)

## What stays the same as the old `run_batch.py`

- Single-source mode still works if you pass one `--data` file. Defaults
  for `--shard-id`/`--num-shards` (0 and 1) make it a no-op.
- The `--workers` flag still controls the save-thread pool (unchanged).
- All physics toggles (`--intrinsic`, `--electronics`, `--no-track-hits`,
  `--sce`, etc.) behave as before.
- HDF5 schemas for sensor/step/hits are unchanged.

## Recovery from interrupted runs

If a shard dies mid-run:

- Outputs for completed source files are intact (each source writes its
  own HDF5 files atomically once finished).
- To resume, re-run with the same `--shard-id`/`--num-shards`. Files
  that already exist in `sensor/run_<id>/` will be **overwritten** (the
  current behavior — there is no resume-skip yet). For now, the easiest
  way to skip is `--file-range` pointing at the unprocessed portion of
  the shard.

## Timing rough numbers

For the doraemon wire dataset (total_pad=1.33M, max_keys=7M, hits_chunk=133k):

- Typical 889k-deposit event: ~1.4 s GPU sim, ~0.4 s load (vlen decode),
  ~0.3 s save → ~2 s/event.
- 100k events serial on one A100: ~55 hours.
- 4 shards × 25k events each: ~14 hours.

If your `n_deposits` distribution is tail-heavy, tail events at 1.3M
deposits cost ~2.5–3 s sim each.
