# Batch pipeline

`production/run_batch.py` is the batch driver: it reads particle step data,
runs the GPU simulation event by event, and writes the structured HDF5 output
described in [data formats](data-formats.md). It is built for throughput —
loading, simulating, and saving all overlap across threads so the GPU stays
busy.

The pipeline is installed as a console script, so these two invocations are
equivalent:

```bash
jaxtpc-batch --data events.h5 --events 100 --dataset myrun --outdir output/
python3 production/run_batch.py --data events.h5 --events 100 --dataset myrun --outdir output/
```

Run from the repository root (the script inserts the repo root on `sys.path`
so `tools/` and `production/` import cleanly).

## The recommended workflow

Sizing the simulator by hand is error-prone: `total_pad`, `max_keys`, the
`fori_loop` chunk sizes, and the box `maxg` all have to be large enough for the
densest event in the dataset or that event crashes or gets logged and skipped
(see [capacities](../concepts/capacities.md)). The profiler measures the data
and emits a config so you don't have to guess:

```bash
# 1. Profile the data (CPU scan + a short GPU probe) → a production config YAML
python3 -m profiler.setup_production \
    --data events.h5 \
    --config config/cubic_wireplane_config.yaml \
    -o config/production_cubic_wireplane_config.yaml

# 2. Run the batch with that config
jaxtpc-batch \
    --data events.h5 \
    --config config/cubic_wireplane_config.yaml \
    --production-config config/production_cubic_wireplane_config.yaml \
    --dataset myrun --outdir output/
```

`--production-config` fills every capacity/threshold field that you did **not**
pass explicitly on the CLI. Precedence is: **explicit CLI flag > production
config > argparse default** — so you can load a config and still override one
knob for an experiment. See [the profiler page](profiler.md) for what each
script measures.

## CLI flags

The flags below are the ones that matter for a production run. Run
`jaxtpc-batch --help` for the complete list.

### Input, output, selection

| Flag | Default | Meaning |
|---|---|---|
| `--data` | `mpvmpr_20.h5` | Input HDF5 file(s), directory(ies), or glob(s). Multiple sources are processed sequentially in one process. |
| `--config` | `config/cubic_wireplane_config.yaml` | Detector geometry/physics YAML. |
| `--production-config` | none | Load profiler-tuned capacities + thresholds. |
| `--dataset` | `sim` | Filename prefix for all output files. |
| `--outdir` | `.` | Output root; `sensor/`, `step/`, `hits/`, `logs/` subdirs are created under it. |
| `--events` | all | Cap the number of events per source file. |
| `--events-per-file` | 1000 | Events per output HDF5 file (splits large sources). |
| `--codec` | `blosc-zstd` | Output compression codec (see note below). |

### Physics toggles

The defaults are: intrinsic noise **off**, coherent noise **off**, electronics
**off**, digitization **on**, track hits **on**.

| Flag | Effect |
|---|---|
| `--intrinsic` | Enable intrinsic (electronics ENC) noise. |
| `--coherent` | Enable coherent, per-group noise (numpy, off-JIT). |
| `--electronics` | Enable the RC⊗RC electronics response. |
| `--no-digitize` | Disable ADC digitization (sensor `values` stay `float32` ADC). |
| `--no-track-hits` | Skip the per-particle decomposition (no `hits/` file). Ignored for pixel readout, which requires track hits. |
| `--sce PATH` | Apply space-charge E-field distortions from an SCE map. |

### Capacities & thresholds

Prefer setting these through `--production-config`. See
[capacities](../concepts/capacities.md) for the overflow semantics.

| Flag | Default | Meaning |
|---|---|---|
| `--total-pad` | 500,000 | Max deposits per volume (sets the compiled shape). |
| `--response-chunk` | 50,000 | Deposits per response `fori_loop` batch; must divide `total-pad`. |
| `--hits-chunk` | 25,000 | Deposits per track-hits `fori_loop` batch; must divide `total-pad`. |
| `--max-keys` | 4,000,000 | Track-hits cell budget per plane. |
| `--maxg` | 200,000 | Box path: group-bucket capacity per event-volume. |
| `--maxg-medium` | none | Enable tiered routing (see below). |
| `--inter-thresh` | 1.0 | In-JIT track-hits pruning threshold. Units: ENC (wire) / ADC (pixel). |
| `--hits-threshold` | 1.0 → 25.0 (wire) | Charge cut for `hits/` CSR entries (YAML key `corr_threshold`). Wire default is bumped to 25.0 ENC. |
| `--threshold-adc` | 2.0 | Sparse sensor threshold (ADC). |
| `--bucketed` | off | Wire-only bucketed accumulation (memory saver). |
| `--max-buckets` | 1,000 | Active buckets per plane in bucketed mode. |

!!! note "Units differ by readout"
    `--inter-thresh` and `--hits-threshold` mean **ENC (electrons)** for wire
    readout and **ADC** for pixel readout — the wire kernel is a dimensionless
    field-impulse fraction applied before electronics, while the pixel kernel
    bakes in the chip gain. See [units](../physics/units.md).

### Threading

| Flag | Default | Meaning |
|---|---|---|
| `--workers` | 2 | Save worker threads (0 = serial save on the main thread). |
| `--read-workers` | 1 | Parallel reader/prefetch threads. |
| `--per-worker-files` | off | Each save worker writes its own output file set (see below). |

### Scale & resume

Covered on the [scaling page](scaling.md): `--shard-id`, `--num-shards`,
`--file-range`, `--skip-existing`.

!!! note "Output codec"
    All datasets are compressed with `--codec` (default `blosc-zstd`, which is
    smaller than gzip **and** faster to read/write). **Reading any non-gzip
    output requires `import hdf5plugin`** — `production/load.py` registers it
    automatically, but ad-hoc `h5py` readers must import it themselves. A
    missing `hdf5plugin` makes the writer raise rather than silently fall back.

## What happens per event

For each event the driver:

1. **Loads** step data and builds a padded, grouped `DepositData` (deposits
   split into volumes, grouped into runs of `--group-size` consecutive steps
   per track, split on spatial gaps of `--gap-threshold` mm).
2. **Simulates** the detector response on the GPU via
   `DetectorSimulator.process_event` — recombination, drift + lifetime
   attenuation, diffusion-convolved wire/pixel response, and (optionally)
   electronics, noise, digitization, and track-hit correspondence.
3. **Saves** three file types — `sensor/`, `step/`, and (with track hits)
   `hits/` — with the dense→sparse conversion and CSR encoding done off the
   main thread.

Output is organized into per-run subfolders when a run id can be resolved from
the source path, e.g. `output/sensor/run_0000000042/myrun_sensor_0000.h5`;
otherwise `run_unknown/` is used.

## Threaded save architecture

The driver runs three roles concurrently so the GPU is the only bottleneck:

```text
Reader threads  →  Main thread          →  Save workers
(prefetch load)    (dispatch GPU sim)      (to_sparse + CSR encode + HDF5 write)

read_q ──▶ main loop ──▶ save_queue ──▶ worker 1 ─┐
                                    └─▶ worker 2 ─┤─▶ HDF5 (per-file locks)
                                    └─▶ worker N ─┘
```

- **Reader prefetch.** `--read-workers` reader threads, each with its own HDF5
  handle, load and *group* events round-robin into a bounded `read_q` while the
  GPU runs. Grouping (`build_deposit_data`) is numpy and releases the GIL, so
  extra readers keep the main loop from going load-bound on dense events.
- **Main dispatch.** The main loop pulls the next prefetched `DepositData`,
  calls `process_event`, and `block_until_ready`s the result (per-event GPU
  sync) before handing the *raw device result* to a save worker via a bounded
  `save_queue`. The expensive host pull (`to_sparse`) happens on the worker,
  not the main thread.
- **Save workers.** `--workers` threads pull from `save_queue`, run
  `to_sparse` + CSR encoding (numpy, GIL-free — these parallelize), and write
  the HDF5.

!!! warning "Per-file locks, not a single lock"
    Earlier docs claimed a single file lock serialized all writes. The current
    code uses **three separate locks** — `sen_lock`, `step_lock`, `hits_lock` —
    one per output file. This lets one worker write the sensor file while
    another writes step or hits, raising the write ceiling from
    `1/(sensor+step+hits)` to `1/max(sensor, step, hits)`. Every worker takes
    the locks in the same order (sensor → step → hits) and holds at most one at
    a time, so there is no deadlock.

**`--per-worker-files`.** With many CPU workers on a single GPU, even per-file
locks can serialize the HDF5 writes. `--per-worker-files` gives each save
worker its **own** output file set (suffixed `_wNN`) and its own uncontended
lock trio, so writes run on separate cores. Output becomes N file sets per
source that downstream readers consume as a group. Use it when the GPU is the
ceiling and the save can't keep up otherwise.

!!! tip "Diagnose the pipeline"
    Set `JAXTPC_PROFILE_SAVE=1` to print per-worker phase timings (post/encode/
    lockwait/write-sensor/write-step/write-hits) plus the save-queue depth from
    the main loop — this tells you whether you are sim-bound, load-bound, or
    save-bound.

## Tiered routing (optional)

Simulation cost grows with the box capacity `maxg`, so sizing a single `maxg`
for the densest tail event makes every small event pay the tail's price.
`--maxg-medium` builds **two** simulators — a faster *medium* tier at
`--maxg-medium` and a *high* tier at `--maxg` — and routes each event by its
**exact** group count (known for free on the host after load):

```python
max_ng = max(len(g2t) for g2t in deposits.group_to_track)
active_sim = sim_high if max_ng > maxg_medium else sim_medium
```

Only events whose largest volume exceeds `maxg_medium` pay the high-tier cost;
the rest use the medium sim. Both tiers share the GPU sequentially (only one
runs at a time), and the profiler can emit both a medium (p99) and a high (max
+ margin) `maxg`. Tiered routing requires the box track-hits path (the default;
disabled by `--no-box`).

## See also

- [Capacities & overflow](../concepts/capacities.md) — sizing `total_pad`,
  `max_keys`, `maxg`, and the overflow failure modes.
- [Data formats](data-formats.md) — the on-disk sensor/step/hits/labl schema
  and decode recipes.
- [Profiler](profiler.md) — how the production config is generated.
- [Scaling](scaling.md) — sharding, resume, and running across many GPUs.
