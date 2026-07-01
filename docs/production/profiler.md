# Profiler & capacity sizing

The production simulator compiles **once** for fixed array shapes, so every
buffer it allocates has a statically sized capacity that must be large enough for
the data. Under-size one and the sim fails at runtime — from a hard
`RuntimeError` to a log-and-reprocess, depending on which capacity overflowed
(see [capacities](../concepts/capacities.md) for the canonical overflow
reference). The `profiler/` tools scan the data and benchmark the sim to size the
whole set and emit a **production config** YAML that `run_batch --production-config`
consumes.

!!! tip "Just want it sized?"
    ```bash
    python3 -m profiler.setup_production --data <run_dir/> \
        --config <detector.yaml> -o <production.yaml>
    python3 production/run_batch.py --data events.h5 \
        --config <detector.yaml> --production-config <production.yaml>
    ```
    `--data` accepts files, directories (globbed `*.h5`), or several of each.
    The rest of this page explains what each tool sizes and how.

## What gets sized

| Parameter | Controls | Set by |
|---|---|---|
| `total_pad` | max deposits per volume (JIT shape) | Step 1 scan |
| `maxg` / `maxg_medium` | group-bucket capacity / tiered-routing split | Step 1 / Step 3 |
| `box_bpy/bpz/bt` (pixel), `box_bw/btw` (wire) | per-group footprint dims | Step 1 (analytic) |
| `max_keys` | track-hits box-cell budget | Step 1 **charge-aware** estimate |
| `response_chunk` / `hits_chunk` | `fori_loop` batch sizes (speed) | Step 2 GPU timing |
| `inter_thresh` | in-JIT box-cell threshold (drives `max_keys`) | fixed 1.0 |
| `threshold_adc` / `corr_threshold` | output thresholds | Step 4 (`--run-thresholds`) |

Track-hits uses the **box** path by default
(`create_track_hits_config(box_enabled=True)`).

---

## `setup_production` — one-shot generator

`profiler/setup_production.py` orchestrates the whole run in four steps. Only
steps 2–4 need the GPU; step 1 is a single CPU pass (parallel `--workers`).

| Step | Output | Cost |
|---|---|---|
| 1. Combined scan | `total_pad`, `maxg`, box dims, **`max_keys`** | CPU |
| 2. Chunk optimization | `response_chunk`, `hits_chunk` | GPU (one JIT per candidate) |
| 3. maxg benchmark | `maxg_medium` (tiered-routing split) | GPU |
| 4. Threshold analysis (`--run-thresholds`) | `corr_threshold`, `threshold_adc` | GPU |

The step-1 scan yields, in one pass: deposit counts → `total_pad` (max, or
p99.9 with `--use-p999`); the n_groups distribution → `maxg` (p99.95,
readout-independent); per-group footprint extents → box dims (analytic); and the
per-deposit **charge-aware** key estimate → `max_keys`.

```bash
python3 -m profiler.setup_production --data run_dir/ --config config.yaml -o out.yaml
python3 -m profiler.setup_production ... --headroom 1.1 --cstar 2.5 --divisor 1
python3 -m profiler.setup_production ... --run-thresholds   # also calibrate thresholds
python3 -m profiler.setup_production ... --skip-chunks      # reuse defaults, skip GPU step 2
```

---

## The `max_keys` estimate (charge-aware)

`max_keys` is the trickiest capacity because it depends on **charge**, not just
geometry. It is the number of per-group box cells whose accumulated
`|signal| > inter_thresh` across the event (`tools/track_hits.py`, box path).

The profiler estimates it **without simulating** each event
(`profiler/estimate_max_keys.py`): for every deposit it counts the
response-kernel cells that clear the *absolute* `inter_thresh` given that
deposit's intensity (recombination × drift attenuation), and sums. A naive
charge-independent geometry count (kernel cells above ~0.5% of peak)
**under-counts ~3×** because it ignores the deposit's brightness; the
charge-aware count fixes that.

The per-deposit sum then **over-counts** the within-group union by a per-readout
overlap factor, corrected by two calibrated knobs (`--cstar`, `--divisor`):

| Readout | Knob (default) | Why |
|---|---|---|
| **pixel** | `c* = 2.5` (threshold ×) | overlap lives in the kernel *tails* — a higher threshold removes it |
| **wire** | `÷ 3.79` (flat factor) | overlap is *structural* (1-D wire projection × 3 planes); the threshold plateaus, so a flat factor is used |

Defaults are readout-aware and calibrated on a reference dataset. **Re-verify
for very different data with `compare_max_keys`** (sweep the threshold/knobs
until `EST/ACTUAL ≈ 1.0`).

!!! danger "`max_keys` overflow is a hard crash — the count tells you the fix"
    If the estimate is too low, `process_event` raises at runtime:
    ```text
    track_hits overflow vol 0 plane 2: count=9,412,880 >= max_keys=9,000,000.
    Increase --max-keys or run profiler.setup_production.
    ```
    This is a hard `RuntimeError`, **not** a silent truncate-and-continue. The
    reported `count` is the *true* cell count, so it tells you exactly how high
    to set `max_keys`. The sim refuses to ship a truncated result. See
    [capacities → `max_keys`](../concepts/capacities.md#max_keys-track-hits-box-cell-budget).

---

## Individual tools

Use these to size one parameter at a time, or to validate `setup_production`'s
output.

| Tool | Sizes | Cost |
|---|---|---|
| `find_optimal_pad` | `total_pad` from deposit-count scan | CPU |
| `find_optimal_maxg` | `maxg` + box dims + charge-aware `max_keys` in one scan | CPU |
| `estimate_max_keys` | the charge-aware `max_keys` estimator (value tables + charge model) | CPU |
| `compare_max_keys` | validate/calibrate the estimate vs the **actual box sim** | GPU |
| `scan_values` | step-1 values (+ plots) patched into an existing config | CPU |
| `find_optimal_chunks` | `response_chunk`, `hits_chunk` via divisor timing | GPU |
| `threshold_analysis` | `corr_threshold`, `threshold_adc` from post-sim sweeps | GPU |

### `find_optimal_pad`
Scans deposit counts across the dataset → `total_pad` (no sim).

### `find_optimal_maxg`
One CPU scan yielding `maxg` + box dims + the charge-aware `max_keys` (pass
`value_tables` + `charge_model` + `key_thresh` for the charge-aware path).

### `estimate_max_keys`
The charge-aware estimator itself (value tables + charge model) that
`find_optimal_maxg` / `scan_values` call.

### `compare_max_keys` — validate / calibrate (GPU)
Runs the **actual box sim** on the top-K densest events and compares to the
geometry estimate. Use it to confirm `max_keys` won't overflow and to calibrate
`c*`/`divisor` for new data (sweep the threshold until `EST/ACTUAL ≈ 1.0`).

```bash
python3 -m profiler.compare_max_keys --data run_dir/ --config config/cubic_pixel_config.yaml \
    --rank-validate --top-k 50 --maxg 130000 --box-bpy 8 --box-bpz 8 --box-bt 83 \
    --total-pad 450000 --hits-chunk 28125 --probe-max-keys 12000000
```

Reports per-event `PERDEP`/`CHG` estimate vs `ACTUAL` and their ratios.
(`--no-box` for the legacy merge path.)

### `scan_values` — CPU-only values + plots
Runs just step 1 (charge-aware `max_keys` + `maxg` + box dims + `total_pad`) over
all files and **patches an existing config**, keeping its GPU-derived chunks +
`maxg_medium`. Saves distribution plots to `profiler/figures/`. Use when only the
**data** changed (not the sim/geometry).

```bash
python3 profiler/scan_values.py --config config/cubic_pixel_config.yaml --data run_dir/ \
    --existing config/production_pixel.yaml --out config/production_pixel.yaml \
    --tag pixel --cstar 2.5 --divisor 1 --headroom 1.1
# wire: --cstar 1 --divisor 3.79
```

### `find_optimal_chunks` — chunk sizes (GPU)
Divisor search timed on the GPU → `response_chunk` (track-hits off) and
`hits_chunk` (track-hits on). Both **must divide `total_pad` evenly** (enforced
at simulator construction).

### `threshold_analysis` — output thresholds (GPU)
One sim, then post-process sweeps → `corr_threshold` (hits charge loss) and
`threshold_adc` (sparse signal loss). Keeps the largest threshold that loses
≤1%.

Auxiliary probes: `sweep_chunks`/`sweep_chunks_2d`, `bench_fit` (GPU
timing/peak-memory).

---

## Production config format

`setup_production` emits a YAML that `run_batch --production-config` loads:

```yaml
detector_config: config/cubic_pixel_config.yaml
total_pad: 450000
response_chunk: 28125
hits_chunk: 28125
max_keys: 9000000
maxg: 110000          # group-bucket capacity (p99.95; rare overflow → reprocess)
maxg_medium: 50000    # tiered routing: medium-tier maxg
box_bpy: 8            # pixel per-group footprint dims  (wire: box_bw, box_btw)
box_bpz: 8
box_bt: 83
inter_thresh: 1.0     # in-JIT box cell threshold   (ENC wire / ADC pixel)
threshold_adc: 2.0    # sparse sensor output cutoff (ADC, both readouts)
corr_threshold: 25.0  # hits CSR cutoff             (ENC wire / ADC pixel)
max_buckets: 1000     # bucketed tile capacity (unused for pixel)
```

!!! warning "The hits cutoff key is `corr_threshold`, not `hits_threshold`"
    Older docs referred to a `hits_threshold` YAML key. The real
    production-config key is **`corr_threshold`** — the charge cutoff applied
    when CSR-encoding the hits file. Groups whose peak falls below it write no
    plane entries (`production/save.py`).

## Thresholds differ by readout

- `inter_thresh` and `corr_threshold` are in **ENC (electrons)** for wire but
  **ADC** for pixel.
- `threshold_adc` is **ADC** for both.

This is by design — the same numeric `inter_thresh` implies a different physical
cut (and thus a different `max_keys` budget) on wire vs pixel. See
[units](../physics/units.md) for the full table.

## Overflow summary

| Capacity | Overflow path |
|---|---|
| `total_pad` | **crash** — `RuntimeError` at load |
| `maxg` | **log & reprocess** — sized at p99.95, rare tail reprocessed at higher `maxg` |
| `max_keys` | **crash** — hard `RuntimeError` (reported count is correct → tells you the fix) |
| `max_buckets` | **crash** — bucketed wire mode only |
| `response_chunk` / `hits_chunk` | **crash at construction** — must divide `total_pad` |

[Capacities & overflow](../concepts/capacities.md) is the canonical reference
for the array each bounds and the verbatim error messages.
