# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

JAXTPC is a GPU-accelerated physics simulation framework for modeling liquid argon Time Projection Chambers (TPCs) used in neutrino physics experiments. It simulates the full detector response chain: charge recombination, electron drift with lifetime attenuation, diffusion-convolved wire/pixel response via Gaussian-blurred kernel interpolation, optional electronics shaping, noise injection, and ADC digitization. Supports arbitrary multi-volume detector geometries (SBND, MicroBooNE, ICARUS, DUNE FD1, DUNE ND-LAr) with both wire and pixel readout. The framework supports both production batch processing and a differentiable path for gradient-based optimization.

## Repository Structure

```
JAXTPC/
├── tools/                     # Core simulation library
│   ├── simulation.py          # DetectorSimulator class (two paths: production + differentiable)
│   ├── config.py              # NamedTuple parameter bundles (SimConfig, SimParams, DepositData, etc.)
│   ├── geometry.py            # YAML config parser → per-volume geometry computation
│   ├── physics.py             # Shared physics pipeline (compute_volume_physics, compute_plane_physics)
│   ├── drift.py               # JIT-compiled drift physics (distance, time, SCE corrections)
│   ├── recombination.py       # Charge/light calculation (Modified Box + EMB models)
│   ├── wires.py               # Wire geometry, deposit preparation, dense/bucketed accumulation
│   ├── kernels.py             # Response kernel loading, Gaussian diffusion table, runtime interpolation
│   ├── electronics.py         # RC⊗RC electronics shaping via sparse FFT
│   ├── noise.py               # MicroBooNE noise model (ENC from wire length)
│   ├── coherent_noise.py      # Tagged coherent (per-wire-group) noise model
│   ├── track_hits.py          # Track hit labeling (group-based charge attribution)
│   ├── efield_distortions.py  # Space charge effects (SCE maps, trilinear interpolation)
│   ├── loader.py              # HDF5 I/O, volume splitting, group ID assignment, padding
│   ├── output.py              # Output format conversion (dense ↔ sparse ↔ bucketed)
│   ├── visualization.py       # Multi-plane plotting (dense + sparse, DeadbandNorm)
│   ├── pixel_visualization.py # Pixel-readout signal visualization (sparse/dense projections)
│   ├── particle_generator.py  # Differentiable muon track generation (PDG dE/dx tables)
│   ├── losses.py              # Multi-scale spectral blur MSE loss for optimization
│   ├── nn_utils.py            # NN inference utilities (symlog, kernel unfolding)
│   ├── sparse_utils.py        # Dense ↔ truly sparse format conversion
│   ├── utils.py               # Standalone HDF5 event I/O (save_event/load_event)
│   ├── responses/             # Pre-computed wire response kernels (NPZ per plane type)
│   └── data/                  # PDG muon dE/dx table
├── production/                # Batch processing pipeline
│   ├── run_batch.py           # CLI batch simulator → structured HDF5 output
│   ├── save.py                # HDF5 writers (sensor/step/hits with delta + CSR encoding)
│   ├── load.py                # HDF5 readers + minimal viz config builder
│   ├── make_labl.py           # Separate writer: per-track labl/ files (stand-in)
│   └── view_production.ipynb  # Visualize production output (no sim needed)
├── profiler/                  # Production parameter auto-tuning
│   ├── setup_production.py    # One-shot pad/maxg/max_keys/chunks/thresholds config
│   ├── find_optimal_pad.py    # Scan data → total_pad
│   ├── find_optimal_maxg.py   # One CPU scan → maxg + max_keys
│   ├── estimate_max_keys.py   # Charge-aware max_keys estimator (value tables + charge model)
│   ├── compare_max_keys.py    # Validate/calibrate estimate vs actual box sim (GPU)
│   ├── scan_values.py         # CPU-only values + config patch + plots
│   ├── find_optimal_chunks.py # Two-pass timing → response_chunk, hits_chunk
│   └── threshold_analysis.py  # Post-sim sweep → threshold_adc, corr_threshold
├── tests/                     # Pytest suite (264 tests, CPU-only, synthetic data)
│   └── conftest.py            # Fixtures: jax_key, minimal_detector_config, ...
├── viewer/                    # Interactive 3D/2D HTML viewer + GIF export
│   ├── serve_viewer.py        # Local HTTP server with byte-range HDF5 support
│   └── export_gif.py          # Standalone rotating 3D GIF/MP4 generator
├── config/                    # Detector configurations
│   ├── cubic_wireplane_config.yaml  # Default: dual-TPC, SBND-scale, U/V/Y planes
│   ├── sbnd_config.yaml, microboone_config.yaml, icarus_config.yaml,
│   │   dune_ndlar_config.yaml (70 volumes), dune_fd1_config.yaml,
│   │   cubic_pixel_config.yaml (pixel readout: same geometry as cubic_wireplane, 1000×1000 pixels/volume)
│   ├── noise_spectrum.npz     # Empirical noise spectral shape
│   └── sce_jaxtpc.h5          # Space charge effect correction maps
├── notebooks/                 # Themed example notebooks (see notebooks/README.md)
│   ├── getting_started/       # quickstart + wire_simulation
│   ├── physics/               # response_kernels (+ planned recombination/diffusion/SCE)
│   ├── readout/               # pixel_simulation, wire-vs-pixel/units
│   ├── gradients/ reco/ calibration/   # differentiable, reconstruction, calibration
│   └── production/            # view_production (+ planned batch/profiler)
```

## Core Architecture

### DetectorSimulator (`tools/simulation.py`)

Central class with two execution paths:

- **`process_event(deposits, key)`** — Production path. Uses `jax.lax.fori_loop` for batched response accumulation with bounded peak memory. Supports noise, electronics, digitization, and track labeling inside a single JIT function. Returns `(response_signals, track_hits_raw, deposits_with_charge)`.
- **`forward(params, deposits)`** — Differentiable path. Uses `jax.remat` for memory-efficient reverse-mode gradients through all physics parameters (velocity, lifetime, diffusion, recombination). Requires `differentiable=True, n_segments=N` at construction.
- **`forward_segments(params, positions_mm, de, dx)`** — Lightweight differentiable forward for segment-like data; masks volumes by position range (no numpy splitting, fully traceable).
- **`process_event_light(deposits)`** — Compute per-segment charge and scintillation photons only (no wire response).

Construction builds per-volume closures for SCE, response, electronics, noise, digitization, and track hits. These are unrolled at trace time (volume/plane loops), so `(vol_idx, plane_idx)` dict lookups work inside JIT. Volumes with zero deposits contribute nothing because the single padding mask in `compute_volume_physics` zeroes all charges when `n_actual=0` (all volumes run with uniform shapes; there is no conditional skip).

### Multi-Volume Architecture

The detector is defined as N independent volumes in YAML. Each volume has its own:
- Spatial range, drift direction (+1 or -1), anode position
- Readout planes: wire (U/V/Y with independent angles/spacings/counts) or pixel
- Diffusion parameters derived from max drift distance
- Response kernels, noise model, electronics, SCE maps

Deposits are split by x-position into volumes during loading (`build_deposit_data`), padded to a fixed `total_pad` per volume for stable JIT shapes.

**Local coordinates.** The loader transforms deposits to volume-local frame (`x_local = drift_direction * (x_anode - x_global)` ≥ 0; y/z centered). All volumes share reference geometry in local frame, so the JIT body uses fixed constants — no per-volume geometry indexing inside the scan. Sensor file save applies the inverse transform before writing.

**Volume iteration.** Controlled by `iterate_mode` at simulator construction: `'scan'` (default, `lax.scan`) or `'vmap'`. One compiled body handles any N volumes.

### Configuration System

Two parameter bundles control the simulation:

- **`SimConfig`** (static, closure-captured) — Array dimensions, mode flags, volume geometry, plane names. Changing any value requires JIT recompilation.
- **`SimParams`** (dynamic, JIT argument) — Physics scalars (velocity, lifetime, diffusion coefficients, recombination parameters) and optional NN/SCE models. Can be changed without recompilation.

Detector geometry is defined in YAML (`config/cubic_wireplane_config.yaml`):
```yaml
volumes:
  - id: 0
    geometry:
      ranges: [[-216.0, 0.0], [-216.0, 216.0], [-216.0, 216.0]]  # cm [x, y, z]
      drift_direction: -1
    planes:
      - {plane_id: 0, angle: 60.0, wire_spacing: 0.3, distance_from_anode: 0.6}
      - {plane_id: 1, angle: -60.0, wire_spacing: 0.3, distance_from_anode: 0.3}
      - {plane_id: 2, angle: 0.0, wire_spacing: 0.3, distance_from_anode: 0.0}
simulation:
  drift: {velocity: 1.6, longitudinal_diffusion: 7.2, transverse_diffusion: 12.0, electron_lifetime: 10.0}
  charge_recombination:
    model: emb   # 'modified_box' or 'emb'
    recomb_parameters: {alpha: 0.93, beta: 0.212, alpha_emb: 0.904, beta_90: 0.204, R_anisotropy: 1.25}
readout:
  sampling_rate: 2.0         # MHz
  electrons_per_adc: 182
electric_field:
  field_strength: 500.0      # V/cm
```

Multiple detector configs available: SBND, MicroBooNE, ICARUS, DUNE FD1, DUNE ND-LAr.

### Data Types

- **`DepositData`** — Multi-volume container: `volumes` (tuple of `VolumeDeposits`), `group_to_track` (numpy lookups per volume, outside JIT), `original_indices`.
- **`VolumeDeposits`** — Single-volume padded arrays: `positions_mm (N,3)`, `de`, `dx`, `theta`, `phi`, `track_ids`, `group_ids`, `t0_us`, `interaction_ids`, `root_track_ids`, `pdg`, `charge`, `photons`, `qs_fractions`, `n_actual`. Padding entries have `de=0, dx=1, track_ids=-1`.
- **`VolumeIntermediates`** — Output of `compute_volume_physics`: charges (zeroed for padding), photons, drift distance/time, positions.
- **`PlaneIntermediates`** — Output of `compute_plane_physics`: per-plane drift, tick time, attenuation, wire indices, charges (zeroed outside readout window).

## Physics Pipeline

For each event, per volume, per plane:

1. **Recombination** (`recombination.py`): Energy deposits → ionization electrons (Q) + scintillation photons (L). Two models via `compute_quanta()`:
   - *Modified Box* (ArgoNeuT): ξ = β/(ρ·E) · dE/dx — no angular dependence
   - *EMB* (ICARUS 2024): adds angular correction β_eff(φ) — tracks parallel to E-field recombine more
   - Both share: R = ln(max(α + ξ, 1)) / ξ; Q = N_i × R; L = ΔE/W_ph − Q

2. **Drift** (`drift.py`): Compute drift distance/time to furthest wire plane, then correct per-plane. Optional SCE corrections (E-field distortions + spatial displacement).

3. **Wire geometry** (`wires.py`): Project (y,z) → closest wire index and distance for each plane's angle/spacing.

4. **Response** (`kernels.py`): reflect-pad + separable Gaussian convolution produces a `DKernel` table indexed by diffusion level `s = drift_distance/max_drift`. Runtime: interpolate DKernel at each deposit's s-value, produce `(N, kW, kH)` response contributions.

5. **Accumulation** (`physics.py`): `fori_loop` over chunks of `response_chunk_size` deposits:
   - **Dense mode**: scatter-add `(kW, kH)` kernels into `(num_wires, num_time)` array
   - **Bucketed mode**: scatter into `(max_buckets, B1, B2)` sparse buckets (lower memory)

6. **Post-processing** (optional, inside JIT):
   - **Electronics** (`electronics.py`): RC⊗RC impulse response via sparse FFT on active wires
   - **Noise** (`noise.py`): MicroBooNE model — ENC = sqrt(x² + (y + z·L)²) with empirical spectral shaping
   - **Digitization** (`electronics.py`): ADC quantization with per-plane pedestals (12-bit default)
   - **Track labeling** (`track_hits.py`): Group-based charge attribution using diffusion kernel neighbors

### Units convention (wire ≠ pixel, by design)

The wire and pixel paths use *different* unit conventions at the hits stage. This is intentional: wire applies the field response *pre*-electronics so the kernel is dimensionless, while pixel bakes the chip gain into the kernel and skips the digitize step entirely.

| Readout | Kernel | Hits stage | Sensor signal | Pipeline |
|---|---|---|---|---|
| **Wire** | Dimensionless e-impulse fraction | **ENC** (electrons) | **ADC** (12-bit, post-digitize) | response (ENC) → electronics → noise → digitize → ADC |
| **Pixel** | ADC per drift-electron (gain baked in) | **ADC** | **ADC** (no separate digitize) | response → done (signal and hits derived from same pass) |

**Threshold units by readout** (the ones that bite — same field name, different meaning):

| Threshold | Wire | Pixel | Where applied |
|---|---|---|---|
| `inter_thresh` | ENC | ADC | In JIT: box compaction (`track_hits.py:1144` wire / `:839` pixel) and merge per-chunk pruning (`merge_chunk_sensor_hits:367`) |
| `corr_threshold` / `hits_threshold` | ENC | ADC | Host, CSR encode (`save.py:447` wire / `:528` pixel) |
| `threshold_adc` | ADC | ADC | Host, `to_sparse` (`output.py:155, 203, 250…`) on sensor only |

**Wire kernel NPZ caveat:** `tools/responses/{U,V,Y}_plane_kernel.npz` carry `units = 'ADC_per_electron'` plus an `adc_per_electron ≈ 0.005` field. **In this pipeline `tools/kernels.py:load_response_kernels` treats the kernel values as a dimensionless field-impulse contribution** — `intensity (electrons) × kernel → ENC`. The `adc_per_electron`/`electrons_per_adc` metadata is *not* consumed in the JIT path; it's informational, reflecting the kernel's first-principles calibration source. The "ADC" in `_digitize_signal` is just the quantization step (round + pedestal + clip) on values that the downstream electronics chain has shaped into ADC-scale.

## Output Formats

Three internal formats, converted via `tools/output.py`:
- **dense**: `(num_wires, num_time_steps)` array per plane
- **bucketed**: 5-tuple `(buckets, num_active, compact_to_key, B1, B2)` — from bucketed accumulation
- **wire_sparse**: 3-tuple `(active_signals, wire_indices, n_active)` — from electronics on bucketed

Downstream: `to_dense()` and `to_sparse()` convert any format. Sparse = `{(vol, plane): {'wire', 'time', 'values'}}`.

## Running Simulations

### Interactive (single event)
Main entry point: `notebooks/getting_started/wire_simulation.ipynb`

```python
from tools.simulation import DetectorSimulator
from tools.geometry import generate_detector
from tools.loader import load_event
from tools.config import create_track_hits_config

detector_config = generate_detector('config/cubic_wireplane_config.yaml')
simulator = DetectorSimulator(detector_config, include_track_hits=True, include_digitize=True)
deposits = load_event('data.h5', simulator.config, event_idx=0)
response_signals, track_hits_raw, deposits = simulator.process_event(deposits, key=jax.random.PRNGKey(42))
```

### Batch production
```bash
python3 production/run_batch.py --data events.h5 --events 100 --dataset myrun --outdir output/
python3 production/run_batch.py --data events.h5 --intrinsic --electronics --workers 2

# Recommended: generate optimized config with profiler, then use it
python3 -m profiler.setup_production --data events.h5 --config config/cubic_wireplane_config.yaml
python3 production/run_batch.py --data events.h5 \
    --config config/cubic_wireplane_config.yaml \
    --production-config config/production_cubic_wireplane_config.yaml
```

The profiler scans data (CPU) and benchmarks the sim (GPU, chunks + maxg_medium only) to set `total_pad`, `maxg`/`maxg_medium`, box dims, `max_keys`, `response_chunk`, `hits_chunk`, `threshold_adc`, `corr_threshold`. `max_keys` is a **charge-aware geometry estimate** (`estimate_max_keys.py`): per deposit, count response-kernel cells clearing the absolute `inter_thresh` for that deposit's intensity (recombination × attenuation), summed — then corrected by a per-readout overlap knob (**pixel `c*=2.5`** threshold, **wire `÷3.79`** factor; calibrated on doraemon, verify with `compare_max_keys`). Track-hits is **box mode by default**. Without a profiled config, mis-sized `total_pad`/`max_keys`/`max_buckets` raise `RuntimeError` (or truncate+log) at runtime. See `profiler/README.md`.

Produces three HDF5 file types per batch:
- `{dataset}_sensor_{NNNN}.h5` — sparse thresholded raw readout (delta-encoded, uint16 if digitized)
- `{dataset}_step_{NNNN}.h5` — 3D truth deposits (pure physics: positions + de/dx/theta/phi/t0_us + charge/photons; no instance or track info)
- `{dataset}_hits_{NNNN}.h5` — per-particle sensor decomposition + group machinery (deposit_to_group, qs_fractions, group_to_track per volume; per-plane CSR-encoded pixel entries)

A fourth file, `{dataset}_labl_{NNNN}.h5`, carries per-track labels and the per-deposit → track_id foreign key. It is produced separately via `production/make_labl.py` (temp stand-in; reads hits + edepsim). See `production/README.md` for the labl schema and workflow.

Threaded save architecture: reader prefetch thread loads the next event's HDF5 while GPU sim runs; main thread dispatches sim; save worker threads (default 2) encode CSR + write HDF5 in parallel with per-file locks (`sen_lock`, `step_lock`, `hits_lock`) so sensor/step/hits writes overlap across workers. Enable `JAXTPC_PROFILE_SAVE=1` to see per-phase timings (encode/lockwait/write-sensor/write-step/write-hits) in the save log.

**Output compression** (`--codec`, default `blosc-zstd`): all datasets are compressed with the codec set via `production/save.py:set_codec`. blosc-zstd is smaller than gzip *and* faster on both read and write (gzip is Pareto-dominated). Alternatives: `blosc-lz4hc` (gzip's size, ~4× faster reads), `blosc-lz4` (fastest read+write, +19% size), `gzip`, `lzf`. **Reading non-gzip output requires `import hdf5plugin`** — `production/load.py` and pimm-data's readers register it automatically; ad-hoc `h5py` consumers must import it themselves. Re-encode existing files between codecs with `pimm-data/scripts/transcode_codec.py`.

### Loading production output
```python
from production.load import build_viz_config, load_event_sensor, load_event_step, load_event_hits

viz_config = build_viz_config('output/sensor/sim_sensor_0000.h5')
dense_signals, attrs, pedestals = load_event_sensor(sensor_path, event_idx=0)
volumes = load_event_step(step_path, event_idx=0)
track_hits, truth_dense, g2t = load_event_hits(hits_path, event_idx=0, num_time_steps=2701)
```

### Tests
```bash
# Fast tests only (~2.5 min on CPU)
JAX_PLATFORM_NAME=cpu python3 -m pytest tests/ -v -m "not slow"

# Full suite, includes integration (~13 min on CPU) — requires response kernel NPZ files
JAX_PLATFORM_NAME=cpu python3 -m pytest tests/ -v

# Single test
JAX_PLATFORM_NAME=cpu python3 -m pytest tests/test_recombination.py::test_mip_survival -v
```

All tests run CPU-only on synthetic data. Markers: `slow` (kernel-dependent integration), `requires_config`, `requires_kernels`. Fixtures live in `tests/conftest.py`. See `tests/TESTS.md` for the full module-by-module breakdown.

### Interactive viewer / GIF export
```bash
python3 viewer/serve_viewer.py output/ --open               # browser viewer
python3 viewer/export_gif.py output/step/sim_step_0000.h5 --event 0
```

## Key Technical Patterns

### JAX Integration
- All physics calculations are JIT-compiled for GPU acceleration
- `jax.vmap` for vectorized operations (deposit preparation, response computation)
- `jax.lax.fori_loop` for bounded-memory batched accumulation
- Single `n_actual` padding mask (in `compute_volume_physics`) zeroes padded/empty-volume deposits so all volumes run with uniform JIT shapes
- `jax.remat` for memory-efficient gradients in the differentiable path
- Always call `jax.block_until_ready()` for proper device synchronization

### Performance Patterns
- Fixed `total_pad` per volume ensures a single JIT compilation for all events
- `response_chunk_size` must evenly divide `total_pad`
- Volume/plane loops unrolled at trace time — closures capture static config
- Factory pattern: per-volume functions built at init, captured in JIT closure
- Padding entries have `de=0` → zero charges after recombination → zero contributions everywhere downstream (single masking point in `compute_volume_physics`)

### Tiered production (planned)
Sim cost is linear in capacities (`sim_time ≈ 0.13 + 5.4µs × maxg`). A single conservative `maxg` (sized for the tail) pays worst-case cost on every event. **Tiered routing** builds two `DetectorSimulator` instances (medium + high) with different `maxg`/`total_pad`/`max_keys` and routes per-event based on the **exact n_groups** (free on the host after load via `len(deposits.group_to_track[v])`):
```python
max_ng = max(len(g2t) for g2t in deposits.group_to_track)
sim = sim_high if max_ng >= maxg_medium else sim_medium
```
Per-volume routing is not practical: `lax.scan`/`vmap` require uniform shapes across the volume axis, and all capacities are shared within one simulator. Per-event routing with two simulators requires zero changes to `tools/` — only `run_batch.py` dispatch logic + a second production config YAML. GPU memory is sequential (only one sim runs at a time). Profiler emits both medium (p99) and high (max + margin) configs. Expected ~2× throughput improvement over single-high for the doraemon dataset.

### Segment Correspondence
- Deposits grouped into runs of N consecutive steps per track, split on spatial gaps
- Group IDs computed per-volume (groups never span volumes)
- `qs_fractions`: each deposit's share of its group's recombined charge
- Track labeling derives pixel-level track IDs from group-level correspondence
- `group_to_track` lookup maps group IDs back to Geant4 track IDs (numpy, outside JIT)

## Dependencies

- JAX (GPU computation)
- NumPy (host-side array operations)
- Matplotlib (visualization)
- H5py (HDF5 I/O)
- hdf5plugin (blosc/zstd/lz4 HDF5 filters — required to read production output, whose default codec is blosc-zstd)
- PyYAML (YAML config parsing)

## Development Notes

- Use `python3` (not `python`) on this system
- Tests live in `tests/` (pytest, CPU-only); see the Tests section above for commands
- JIT compilation causes initial warmup; `simulator.warm_up()` triggers it with dummy data
- Memory management important for large events (500k+ deposits per volume) — pixel readout uses the box (group-as-bucket) track-hits path by default; `--bucketed` is an optional wire-only sensor-accumulation memory-saver (not required for pixel). Use the profiler to size capacities (`total_pad`/`max_keys`/`maxg`)
- Response kernels stored as NPZ in `tools/responses/` (U/V/Y plane types)
- Group ids in production `hits/` files are **1-based** (entry `group_to_track[0]` is unused); see `production/README.md` for the full correspondence schema
- Do not add Claude as a `Co-Authored-By` trailer on commits