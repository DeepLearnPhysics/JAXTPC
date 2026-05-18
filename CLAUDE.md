# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

JAXTPC is a GPU-accelerated physics simulation framework for modeling liquid argon Time Projection Chambers (TPCs) used in neutrino physics experiments. It simulates the full detector response chain: charge recombination, electron drift with lifetime attenuation, diffusion-convolved wire response via DCT-based kernel interpolation, optional electronics shaping, noise injection, and ADC digitization. The framework supports both production batch processing and a differentiable path for gradient-based optimization.

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
│   ├── kernels.py             # Response kernel loading, DCT diffusion table, runtime interpolation
│   ├── electronics.py         # RC⊗RC electronics shaping via sparse FFT
│   ├── noise.py               # MicroBooNE noise model (ENC from wire length)
│   ├── track_hits.py          # Track hit labeling (group-based charge attribution)
│   ├── efield_distortions.py  # Space charge effects (SCE maps, trilinear interpolation)
│   ├── loader.py              # HDF5 I/O, volume splitting, group ID assignment, padding
│   ├── output.py              # Output format conversion (dense ↔ sparse ↔ bucketed)
│   ├── visualization.py       # Multi-plane plotting (dense + sparse, DeadbandNorm)
│   ├── particle_generator.py  # Differentiable muon track generation (PDG dE/dx tables)
│   ├── losses.py              # Multi-scale spectral blur MSE loss for optimization
│   ├── pointcloud.py          # Signal → weighted point cloud for OT losses
│   ├── nn_utils.py            # NN inference utilities (symlog, kernel unfolding)
│   ├── sparse_utils.py        # Dense ↔ truly sparse format conversion
│   ├── space_points.py        # Rough 3D reconstruction from wire crossings
│   ├── responses/             # Pre-computed wire response kernels (NPZ per plane type)
│   └── data/                  # PDG muon dE/dx table
├── production/                # Batch processing pipeline
│   ├── run_batch.py           # CLI batch simulator → structured HDF5 output
│   ├── save.py                # HDF5 writers (sensor/edep/hits with delta + CSR encoding)
│   ├── load.py                # HDF5 readers + minimal viz config builder
│   └── view_production.ipynb  # Visualize production output (no sim needed)
├── config/                    # Detector configurations
│   ├── cubic_wireplane_config.yaml  # Default: dual-TPC, SBND-scale, U/V/Y planes
│   ├── sbnd_config.yaml, microboone_config.yaml, icarus_config.yaml, ...
│   ├── noise_spectrum.npz     # Empirical noise spectral shape
│   └── sce_jaxtpc.h5          # Space charge effect correction maps
├── run_simulation.ipynb       # Interactive single-event simulation notebook
└── closure_analysis*/         # Physics closure test notebooks
```

## Core Architecture

### DetectorSimulator (`tools/simulation.py`)

Central class with two execution paths:

- **`process_event(deposits, key)`** — Production path. Uses `jax.lax.fori_loop` for batched response accumulation with bounded peak memory. Supports noise, electronics, digitization, and track labeling inside a single JIT function. Returns `(response_signals, track_hits_raw, deposits_with_charge)`.
- **`forward(params, deposits)`** — Differentiable path. Uses `jax.remat` for memory-efficient reverse-mode gradients through all physics parameters (velocity, lifetime, diffusion, recombination). Requires `differentiable=True, n_segments=N` at construction.
- **`forward_segments(params, positions_mm, de, dx)`** — Lightweight differentiable forward for segment-like data; masks volumes by position range (no numpy splitting, fully traceable).
- **`process_event_light(deposits)`** — Compute per-segment charge and scintillation photons only (no wire response).

Construction builds per-volume closures for SCE, response, electronics, noise, digitization, and track hits. These are unrolled at trace time (volume/plane loops), so `(vol_idx, plane_idx)` dict lookups work inside JIT. Volumes with zero deposits are skipped via `jax.lax.cond`.

### Multi-Volume Architecture

The detector is defined as N independent volumes in YAML. Each volume has its own:
- Spatial range, drift direction (+1 or -1), anode position
- Wire planes (U/V/Y) with independent angles, spacings, wire counts
- Diffusion parameters derived from max drift distance
- Response kernels, noise model, electronics, SCE maps

Deposits are split by x-position into volumes during loading (`build_deposit_data`), padded to a fixed `total_pad` per volume for stable JIT shapes.

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

4. **Response** (`kernels.py`): DCT-domain Gaussian blurring produces a `DKernel` table indexed by diffusion level `s = drift_distance/max_drift`. Runtime: interpolate DKernel at each deposit's s-value, produce `(N, kW, kH)` response contributions.

5. **Accumulation** (`physics.py`): `fori_loop` over chunks of `response_chunk_size` deposits:
   - **Dense mode**: scatter-add `(kW, kH)` kernels into `(num_wires, num_time)` array
   - **Bucketed mode**: scatter into `(max_buckets, B1, B2)` sparse buckets (lower memory)

6. **Post-processing** (optional, inside JIT):
   - **Electronics** (`electronics.py`): RC⊗RC impulse response via sparse FFT on active wires
   - **Noise** (`noise.py`): MicroBooNE model — ENC = sqrt(x² + (y + z·L)²) with empirical spectral shaping
   - **Digitization** (`electronics.py`): ADC quantization with per-plane pedestals (12-bit default)
   - **Track labeling** (`track_hits.py`): Group-based charge attribution using diffusion kernel neighbors

## Output Formats

Three internal formats, converted via `tools/output.py`:
- **dense**: `(num_wires, num_time_steps)` array per plane
- **bucketed**: 5-tuple `(buckets, num_active, compact_to_key, B1, B2)` — from bucketed accumulation
- **wire_sparse**: 3-tuple `(active_signals, wire_indices, n_active)` — from electronics on bucketed

Downstream: `to_dense()` and `to_sparse()` convert any format. Sparse = `{(vol, plane): {'wire', 'time', 'values'}}`.

## Running Simulations

### Interactive (single event)
Main entry point: `run_simulation.ipynb`

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
python3 production/run_batch.py --data events.h5 --noise --electronics --bucketed --workers 2
```

Produces three HDF5 file types per batch:
- `{dataset}_sensor_{NNNN}.h5` — sparse thresholded raw readout (delta-encoded, uint16 if digitized)
- `{dataset}_edep_{NNNN}.h5` — 3D truth deposits (pure physics: positions + de/dx/theta/phi/t0_us + charge/photons; no instance or track info)
- `{dataset}_hits_{NNNN}.h5` — per-particle sensor decomposition + group machinery (deposit_to_group, qs_fractions, group_to_track per volume; per-plane CSR-encoded pixel entries)

A fourth file, `{dataset}_labl_{NNNN}.h5`, carries per-track labels and the per-deposit → track_id foreign key. It is produced separately via `production/make_labl.py` (temp stand-in; reads hits + edepsim). See `production/README.md` for the labl schema and workflow.

Threaded save architecture: main thread runs GPU sim, worker threads encode CSR + write HDF5 in parallel.

### Loading production output
```python
from production.load import build_viz_config, load_event_sensor, load_event_edep, load_event_hits

viz_config = build_viz_config('output/sensor/sim_sensor_0000.h5')
dense_signals, attrs, pedestals = load_event_sensor(sensor_path, event_idx=0)
volumes = load_event_edep(edep_path, event_idx=0)
track_hits, truth_dense, g2t = load_event_hits(hits_path, event_idx=0, num_time_steps=2701)
```

## Key Technical Patterns

### JAX Integration
- All physics calculations are JIT-compiled for GPU acceleration
- `jax.vmap` for vectorized operations (deposit preparation, response computation)
- `jax.lax.fori_loop` for bounded-memory batched accumulation
- `jax.lax.cond` to skip empty volumes without breaking JIT shapes
- `jax.remat` for memory-efficient gradients in the differentiable path
- Always call `jax.block_until_ready()` for proper device synchronization

### Performance Patterns
- Fixed `total_pad` per volume ensures a single JIT compilation for all events
- `response_chunk_size` must evenly divide `total_pad`
- Volume/plane loops unrolled at trace time — closures capture static config
- Factory pattern: per-volume functions built at init, captured in JIT closure
- Padding entries have `de=0` → zero charges after recombination → zero contributions everywhere downstream (single masking point in `compute_volume_physics`)

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
- PyYAML (YAML config parsing)

## Development Notes

- Use `python3` (not `python`) on this system
- No formal test suite — validation via physics closure tests and analysis notebooks
- JIT compilation causes initial warmup; `simulator.warm_up()` triggers it with dummy data
- Memory management important for large events (500k+ deposits per volume)
- Response kernels stored as NPZ in `tools/responses/` (U/V/Y plane types)