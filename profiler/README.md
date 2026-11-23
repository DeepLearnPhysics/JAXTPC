# JAXTPC Profiler

Tools for sizing simulation capacities and generating production configs.

The production sim's capacities (`total_pad`, `max_keys`, `maxg`, box dims, chunk
sizes, thresholds) must be sized for the data — otherwise the sim raises
`RuntimeError` (`total_pad`), reprocesses (`maxg`), or or refuses the event (`max_keys`)
at runtime. The profiler scans the data and benchmarks the sim to set them.

## Quick start

```bash
# One command: scan + estimate capacities + optimize chunks + save config
python3 -m profiler.setup_production --data run_dir/ \
    --config config/cubic_pixel_config.yaml \
    -o config/production_pixel.yaml

# Use it in production
python3 production/run_batch.py --data events.h5 \
    --config config/cubic_pixel_config.yaml \
    --production-config config/production_pixel.yaml
```

`--data` accepts files, directories (globbed `*.h5`), or several of each.

## Profiling your own edep-sim / Geant4 setup

A production config is only valid for the sample it was profiled on. Re-profile
when any of these change:

| You changed | What moves |
|---|---|
| Step length or voxel size | `total_pad`, `maxg`, `max_keys`, and the `max_keys` overlap knobs |
| Energy, generator, pile-up per event | `total_pad`, `maxg`, `max_keys`, `maxg_medium` |
| Detector YAML (pitch, drift, readout window, response) | `max_keys`, box dims, chunks |
| Group definition (`--group-size`, `--gap-threshold`) | `maxg`, box dims; pass the same values to `run_batch` |

The input is the edep-sim HDF5 schema read by `tools/loader.py`
(`pstep/lar_vol`, one entry per event). Convert other Geant4 output to it first;
`x`/`y`/`z` (or `position`) and `dx` in mm, `de` in MeV, `t` in ns, plus `theta`,
`phi`, `track_id`, `pdg` and `ancestor_track_id`. A missing field is filled with a
default (zeros; ones for `track_id`), not reported. Positions must be in the detector YAML's frame: deposits
outside every volume are dropped without a message.

1. **Scan and benchmark** on files spread over the whole sample, not the first
   few (capacities are set by rare large events):
   ```bash
   python3 -m profiler.setup_production --data fileA.h5 fileB.h5 ... \
       --config config/my_detector.yaml -o config/production_my_sample.yaml \
       --workers 24 --headroom 1.1
   ```
2. **Check `max_keys` with the real simulation.** The overlap knobs (pixel
   `c*=2.5`, wire `/3.79`) were calibrated on doraemon and depend on step length:
   ```bash
   python3 -m profiler.compare_max_keys --data fileA.h5 fileB.h5 ... \
       --config config/my_detector.yaml --rank-validate --rank-by estimate --top-k 25 \
       --maxg <maxg> --total-pad <total_pad> --hits-chunk <hits_chunk> \
       --response-chunk <response_chunk> --probe-max-keys <2x estimate>
   ```
   Pass the box dims the simulator prints at start-up (`--box-bpy/-bpz/-bt` or
   `--box-bw/-btw`). Set `max_keys` from the largest ACTUAL plus margin. If
   EST/ACTUAL is far from the knob's value, rerun step 1 with `--cstar`/`--divisor`.
3. **Set `maxg` from the largest group count seen (p100) plus margin.** The
   suggestion is p99.95, so it skips the tail. `maxg_medium` makes this cheap:
   events at or below it run in the smaller simulator.
4. **Check the memory.** Pixel needs `maxg x box_bpy x box_bpz x box_bt x 4`
   bytes, and the cell count must stay below 2^31 (the simulator refuses it
   otherwise).
5. **Review the thresholds.** `--run-thresholds` values are a data-volume choice,
   not a capacity.

## The pipeline (`setup_production`)

| Step | Output | Cost |
|---|---|---|
| 1. Combined scan | `total_pad`, `maxg`, box dims, **`max_keys`** | **CPU** (parallel `--workers`) |
| 2. Chunk optimization | `response_chunk`, `hits_chunk` | GPU (one JIT per candidate) |
| 3. maxg benchmark | `maxg_medium` (tiered-routing split) | GPU |
| 4. Threshold analysis (`--run-thresholds`) | `corr_threshold`, `threshold_adc` | GPU |

**Only steps 2–4 need the GPU.** Step 1 is one CPU pass that yields: deposit counts
→ `total_pad` (max, or `--use-p999`); the n_groups distribution → `maxg` (p99.95,
readout-independent); per-group footprint extents → box dims; and the per-deposit
**charge-aware** key estimate → `max_keys`.

Track-hits uses the **box** path by default (`create_track_hits_config(box_enabled=True)`).

## The `max_keys` estimate (charge-aware)

`max_keys` = the number of per-group box cells whose accumulated `|signal| >
inter_thresh` (`tools/track_hits.py`, box path). The profiler estimates it **without
simulating** each event: for every deposit it counts the response-kernel cells that
clear the *absolute* threshold given that deposit's intensity (recombination × drift
attenuation), and sums (`estimate_max_keys.py`). The cheap, charge-independent
geometry count (kernel cells > 0.5% of peak) under-counts ~3× because it ignores the
deposit's brightness; the charge-aware count fixes that.

The per-deposit sum over-counts the within-group **union** by a per-readout overlap
factor, corrected by two calibrated knobs (`--cstar`, `--divisor`):

| readout | knob (default) | why |
|---|---|---|
| **pixel** | `c* = 2.5` (threshold ×) | overlap lives in the kernel *tails* — a higher threshold removes it |
| **wire** | `÷ 3.79` (flat factor) | overlap is *structural* (1-D wire projection × 3 planes); the threshold plateaus, so a factor is used |

Defaults are readout-aware and calibrated on the doraemon dataset. **Re-verify for very
different data with `compare_max_keys`.**

## Scripts

### `setup_production` — one-shot config generator
```bash
python3 -m profiler.setup_production --data run_dir/ --config config.yaml -o out.yaml
python3 -m profiler.setup_production ... --headroom 1.1 --cstar 2.5 --divisor 1
python3 -m profiler.setup_production ... --run-thresholds   # also calibrate thresholds
python3 -m profiler.setup_production ... --skip-chunks       # reuse defaults, skip GPU Step 2
```

### `compare_max_keys` — validate / calibrate the estimate (GPU)
Runs the **actual box sim** on the top-K densest events and compares to the geometry
estimate. Use it to confirm `max_keys` won't overflow, and to calibrate `c*`/`divisor`
for new data (sweep the threshold until EST/ACTUAL ≈ 1.0).
```bash
python3 -m profiler.compare_max_keys --data run_dir/ --config config/cubic_pixel_config.yaml \
    --rank-validate --top-k 50 --maxg 130000 --box-bpy 8 --box-bpz 8 --box-bt 83 \
    --total-pad 450000 --hits-chunk 28125 --probe-max-keys 12000000
```
Reports per-event `PERDEP`/`CHG` estimate vs `ACTUAL`, and the ratios. (`--no-box` for the
legacy merge path.)

### `scan_values` — CPU-only values + plots (no GPU)
Runs just Step 1 (charge-aware `max_keys` + `maxg` + box dims + `total_pad`) over all
files and **patches an existing config**, keeping its GPU-derived chunks + `maxg_medium`.
Saves distribution plots to `profiler/figures/`. Use when only the **data** changed (not
the sim/geometry) — pixel + wire can run concurrently.
```bash
python3 profiler/scan_values.py --config config/cubic_pixel_config.yaml --data run_dir/ \
    --existing config/production_pixel.yaml --out config/production_pixel.yaml \
    --tag pixel --cstar 2.5 --divisor 1 --headroom 1.1
# wire: --cstar 1 --divisor 3.79
```

### Individual step tools
- **`find_optimal_pad`** — scan deposit counts → `total_pad` (CPU, no sim).
- **`find_optimal_maxg`** — `maxg` + box dims + charge-aware `max_keys` in one CPU scan
  (pass `value_tables` + `charge_model` + `key_thresh` for the charge-aware path).
- **`find_optimal_chunks`** — divisor search → `response_chunk` (track_hits off),
  `hits_chunk` (track_hits on), timed on the GPU.
- **`threshold_analysis`** — one sim, then post-process sweeps → `corr_threshold`
  (hits charge loss), `threshold_adc` (sparse signal loss). Keeps the largest threshold
  losing ≤1%.
- Aux/diagnostic: `sweep_chunks(_2d)`, `bench_fit` (GPU timing/peak-mem probes).

## Production config format

```yaml
detector_config: config/cubic_pixel_config.yaml
total_pad: 450000
response_chunk: 28125
hits_chunk: 28125
max_keys: 9000000
maxg: 110000          # group-bucket capacity (p99.95; rare overflow -> reprocess)
maxg_medium: 50000    # tiered routing: medium-tier maxg
box_bpy: 8            # pixel per-group footprint dims  (wire: box_bw, box_btw)
box_bpz: 8
box_bt: 83
inter_thresh: 1.0     # in-JIT box cell threshold   (ENC wire / ADC pixel)
threshold_adc: 2.0    # sparse sensor output cutoff (ADC, both readouts)
corr_threshold: 25.0  # hits CSR cutoff             (ENC wire / ADC pixel)
max_buckets: 1000     # bucketed tile capacity (unused for pixel)
```

## Parameters

| Parameter | Controls | Set by |
|---|---|---|
| `total_pad` | max deposits/volume (JIT shape) | Step 1 scan |
| `maxg` / `maxg_medium` | group-bucket capacity / tiered split | Step 1 / Step 3 |
| `box_bpy/bpz/bt` (pixel), `box_bw/btw` (wire) | per-group footprint dims | Step 1 |
| `max_keys` | box-cell capacity | Step 1 charge-aware estimate (`c*`/`divisor`) |
| `response_chunk` / `hits_chunk` | fori_loop batch sizes (speed) | Step 2 timing |
| `inter_thresh` | in-JIT box cell threshold | fixed 1.0 (drives `max_keys`) |
| `threshold_adc` / `corr_threshold` | output thresholds | Step 4 (`--run-thresholds`) |

## Units caveat (thresholds differ by readout)

- `inter_thresh`, `corr_threshold` — **ENC** (wire) vs **ADC** (pixel).
- `threshold_adc` — **ADC** for both.

See `CLAUDE.md` "Units convention".

## Overflow protection

- **`total_pad`** — deposits exceed capacity → `RuntimeError` at load.
- **`maxg`** — n_groups exceeds capacity → logged; sized at p99.95 so reprocess the rare tail.
- **`max_keys`** — box-cell count exceeds capacity → `RuntimeError`; `run_batch` logs and
  skips the event. Size with `compare_max_keys` on the tail.

## Contents

```
profiler/
  setup_production.py    # one-shot orchestrator (steps 1-4)
  estimate_max_keys.py   # charge-aware max_keys estimator (value tables + charge model)
  find_optimal_maxg.py   # maxg + box dims + max_keys (one CPU scan)
  find_optimal_pad.py    # total_pad scan
  find_optimal_chunks.py # response_chunk + hits_chunk (GPU timing)
  compare_max_keys.py    # validate/calibrate estimate vs actual box (GPU)
  scan_values.py         # CPU-only values + config patch + plots
  threshold_analysis.py  # corr_threshold + threshold_adc
  sweep_chunks(_2d).py / bench_fit.py  # aux GPU timing/peak-mem probes
  plots.py / production_config.py / timing.py  # helpers
```
