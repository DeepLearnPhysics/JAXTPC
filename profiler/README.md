# JAXTPC Profiler

Tools for finding optimal simulation parameters and generating production configs.

## Quick Start

```bash
# One command: scans data, probes max_keys/max_buckets, finds optimal chunks, saves config
python3 -m profiler.setup_production --data events.h5 --config config/cubic_wireplane_config.yaml

# Use the generated config in production
python3 production/run_batch.py --data events.h5 \
    --config config/cubic_wireplane_config.yaml \
    --production-config config/production_cubic_wireplane_config.yaml
```

## Scripts

### `setup_production` — One-shot production config generator

Runs the full optimization pipeline and saves a YAML config:

1. Scans events for optimal `total_pad`
2. Probes `max_keys` with representative events
3. Probes `max_buckets` (if pixel readout or `--bucketed`)
4. Finds optimal `response_chunk` and `hits_chunk` via coarse/fine search

```bash
python3 -m profiler.setup_production --data events.h5 --config config.yaml
python3 -m profiler.setup_production --data events.h5 --config config.yaml -o config/my_production.yaml
python3 -m profiler.setup_production --data events.h5 --config config.yaml --use-max --probe-events 10
```

### Individual Scripts

Each script optimizes a single parameter and can save to a shared config file via `--save-config`.

#### `find_optimal_pad` — Scan data for `total_pad`

Reads positions from HDF5, splits into volumes by geometry, reports deposit count statistics. No simulation needed.

```bash
python3 -m profiler.find_optimal_pad --data events.h5 --config config.yaml
python3 -m profiler.find_optimal_pad --data events.h5 --config config.yaml --events 100 --save-config config/prod.yaml
```

#### `find_optimal_max_keys` — Probe track-hits capacity

Runs with a large `max_keys`, records actual entry counts per plane, suggests a safe value with headroom.

```bash
python3 -m profiler.find_optimal_max_keys --data events.h5 --config config.yaml
python3 -m profiler.find_optimal_max_keys --data events.h5 --config config.yaml --events 10 --headroom 2.0
```

#### `find_optimal_max_buckets` — Probe bucketed tile capacity

Same approach as `max_keys` but for the bucketed accumulation active tile count. Required for pixel readout.

```bash
python3 -m profiler.find_optimal_max_buckets --data events.h5 --config config/cubic_pixel_config.yaml
python3 -m profiler.find_optimal_max_buckets --data events.h5 --config config.yaml --bucketed
```

#### `find_optimal_chunks` — Two-pass chunk size search

Enumerates divisors of `total_pad`, times each with coarse pass (few iterations), then fine pass on top 3.
Phase 1 finds `response_chunk` (track_hits off), Phase 2 finds `hits_chunk` (track_hits on).

```bash
python3 -m profiler.find_optimal_chunks --data events.h5 --config config.yaml --total-pad 300000
python3 -m profiler.find_optimal_chunks --data events.h5 --config config.yaml --lo 5000 --hi 100000 --n-fine 15
```

#### `find_optimal_inter_thresh` — Smallest safe intermediate pruning threshold

Sweeps `inter_thresh` values and compares charge output against a baseline (inter_thresh=0). Finds the largest value within a charge loss tolerance. Each value requires a JIT recompilation.

```bash
python3 -m profiler.find_optimal_inter_thresh --data events.h5 --config config.yaml
python3 -m profiler.find_optimal_inter_thresh --data events.h5 --config config.yaml --tolerance 0.001
```

#### `threshold_analysis` — Measure charge/signal loss from output thresholds

Runs one simulation, then sweeps thresholds in post-processing (no re-simulation):
- `corr_threshold`: filters correspondence entries, measures charge loss
- `threshold_adc`: filters sparse signal output, measures signal loss

```bash
python3 -m profiler.threshold_analysis --data events.h5 --config config.yaml --events 5
python3 -m profiler.threshold_analysis --data events.h5 --config config.yaml --mode corr
python3 -m profiler.threshold_analysis --data events.h5 --config config.yaml --mode adc --adc-values 0.5 1.0 2.0 5.0
```

#### `benchmark_sim` — Time simulation with different configurations

Sweeps feature combinations (noise, electronics, track_hits, digitize) or a single numeric parameter.

```bash
python3 -m profiler.benchmark_sim --data events.h5 --config config.yaml
python3 -m profiler.benchmark_sim --data events.h5 --config config.yaml --sweep total_pad 100000 200000 500000
python3 -m profiler.benchmark_sim --data events.h5 --config config.yaml --features-only --runs 10
```

## Production Config Format

Generated YAML stored in `config/`:

```yaml
# config/production_cubic_wireplane_config.yaml
detector_config: config/cubic_wireplane_config.yaml
total_pad: 300000
response_chunk: 50000
hits_chunk: 25000
max_keys: 4000000
inter_thresh: 1.0
threshold_adc: 2.0
corr_threshold: 25.0
max_buckets: 50000
```

## Parameters

| Parameter | What it controls | How to optimize |
|---|---|---|
| `total_pad` | Max deposits per volume (JIT shape) | `find_optimal_pad` — data scan, no sim |
| `response_chunk` | Deposits per response fori_loop batch | `find_optimal_chunks` — timing sweep |
| `hits_chunk` | Deposits per track_hits fori_loop batch | `find_optimal_chunks` — timing sweep |
| `max_keys` | Track-hits merge state capacity | `find_optimal_max_keys` — probe actual counts |
| `max_buckets` | Active tile capacity (bucketed mode) | `find_optimal_max_buckets` — probe actual counts |
| `inter_thresh` | Track-hits intermediate pruning | `find_optimal_inter_thresh` — charge loss sweep |
| `threshold_adc` | Sparse signal output threshold | `threshold_analysis` — signal loss sweep |
| `corr_threshold` | Correspondence charge cutoff | `threshold_analysis` — charge loss sweep |

## Recommended Order

```bash
DATA=events.h5
CFG=config/cubic_wireplane_config.yaml
OUT=config/production.yaml

# 1. Data scan (no sim)
python3 -m profiler.find_optimal_pad --data $DATA --config $CFG --save-config $OUT

# 2. Capacity probes (one sim build each)
python3 -m profiler.find_optimal_max_keys --data $DATA --config $CFG --save-config $OUT
python3 -m profiler.find_optimal_max_buckets --data $DATA --config $CFG --save-config $OUT  # pixel/bucketed only

# 3. Chunk timing sweep (multiple sim builds)
python3 -m profiler.find_optimal_chunks --data $DATA --config $CFG --save-config $OUT

# 4. Quality analysis (one sim, post-process sweep)
python3 -m profiler.threshold_analysis --data $DATA --config $CFG --save-config $OUT --save-corr 25 --save-adc 2.0
python3 -m profiler.find_optimal_inter_thresh --data $DATA --config $CFG --save-config $OUT

# Or do it all at once:
python3 -m profiler.setup_production --data $DATA --config $CFG -o $OUT
```

## Overflow Protection

Both `total_pad` and `max_keys` raise `RuntimeError` if exceeded at runtime:

- **total_pad**: deposits in a volume exceed capacity. Raised during data loading.
- **max_keys**: track-hits merge state overflows. Raised after simulation.
- **max_buckets**: active bucket tiles exceed capacity. Raised after simulation.

Using a profiler-generated config avoids all three.

## Contents

```
profiler/
├── __init__.py                      # Package docstring
├── timing.py                        # TimingResult, sync_result, time_function
├── production_config.py             # Save/load/update production config YAML
├── setup_production.py              # One-shot: pad + max_keys + max_buckets + chunks
├── find_optimal_pad.py              # Scan data → total_pad
├── find_optimal_max_keys.py         # Probe → max_keys
├── find_optimal_max_buckets.py      # Probe → max_buckets
├── find_optimal_chunks.py           # Two-pass search → response_chunk, hits_chunk
├── find_optimal_inter_thresh.py     # Charge loss sweep → inter_thresh
├── threshold_analysis.py            # Signal/charge loss sweep → threshold_adc, corr_threshold
├── benchmark_sim.py                 # Feature combo and parameter timing
└── README.md                        # This file
```
