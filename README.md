# JAXTPC

## Overview

JAXTPC is a GPU-accelerated Time Projection Chamber (TPC) simulation framework built with JAX. It models the full detector response chain in liquid argon TPCs: charge recombination, electron drift, diffusion-convolved wire/pixel response, electronics shaping, noise injection, and ADC digitization. Supports arbitrary multi-volume detector geometries (SBND, MicroBooNE, ICARUS, DUNE ND-LAr, DUNE FD1) with both wire and pixel readout.

## Repository Structure

```
JAXTPC/
├── tools/                        # Core simulation package
│   ├── simulation.py             # DetectorSimulator class (scan/vmap volume iteration)
│   ├── config.py                 # Data types (SimParams, SimConfig, VolumeGeometry, ...)
│   ├── physics.py                # Physics pipeline (volume + plane computations)
│   ├── geometry.py               # YAML config parser → per-volume geometry
│   ├── kernels.py                # Diffusion kernel generation (spatial conv) + interpolation
│   ├── drift.py                  # Electron drift physics
│   ├── wires.py                  # Wire/pixel geometry, deposit preparation, accumulation
│   ├── recombination.py          # Charge recombination (Modified Box + EMB models)
│   ├── electronics.py            # RC-RC electronics response via sparse FFT
│   ├── noise.py                  # Intrinsic noise generation (MicroBooNE model)
│   ├── track_hits.py             # Group-based track labeling + Q_s fractions
│   ├── efield_distortions.py     # Space charge effects (SCE maps, trilinear interpolation)
│   ├── loader.py                 # HDF5 I/O, volume splitting, local coord transform
│   ├── output.py                 # Format conversion (dense ↔ sparse ↔ bucketed)
│   ├── visualization.py          # Wire signal / track label / diffused charge plotting
│   ├── particle_generator.py     # Differentiable muon track generation
│   ├── losses.py                 # Sobolev / spectral loss functions
│   ├── coherent_noise.py         # Tagged coherent (per-wire-group) noise model
│   ├── pointcloud.py             # Signal → weighted point cloud (OT losses)
│   ├── space_points.py           # Rough 3D reconstruction from wire crossings
│   ├── nn_utils.py               # NN inference utilities
│   ├── sparse_utils.py           # Dense ↔ truly-sparse format conversion
│   ├── utils.py                  # Misc shared helpers
│   └── responses/                # Pre-computed wire response kernels (U/V/Y NPZ)
├── production/                   # Batch production pipeline
│   ├── run_batch.py              # Batch simulation with threaded save workers
│   ├── save.py                   # HDF5 writers (sensor/step/hits, delta + CSR encoding)
│   ├── load.py                   # HDF5 readers + minimal viz config builder
│   ├── make_labl.py              # Per-track label writer (labl files)
│   ├── view_production.ipynb     # Visualize production output
│   ├── README.md                 # Pipeline docs, CLI flags, output schema
│   └── RUN_PRODUCTION.md         # End-to-end production run guide
├── profiler/                     # Production parameter optimization
│   ├── setup_production.py       # Auto-tune total_pad, chunks, max_keys
│   ├── find_optimal_pad.py       # Scan data for max deposits per volume
│   ├── find_optimal_chunks.py    # Find optimal chunk sizes
│   ├── find_optimal_maxg.py      # Scan → maxg + max_keys (one pass)
│   └── ...                       # Per-parameter tuning scripts
├── tests/                        # Pytest test suite
│   ├── test_pipeline.py          # End-to-end integration tests
│   ├── test_pipeline_forward.py  # Differentiable path tests
│   ├── test_simulation.py        # Simulator unit tests
│   ├── test_electronics.py       # Electronics/noise tests
│   └── ...                       # Per-module tests
├── config/                       # Detector configurations
│   ├── cubic_wireplane_config.yaml   # Default: dual-TPC, SBND-scale
│   ├── sbnd_config.yaml              # SBND
│   ├── microboone_config.yaml        # MicroBooNE
│   ├── icarus_config.yaml            # ICARUS (4 volumes)
│   ├── dune_ndlar_config.yaml        # DUNE ND-LAr (70 volumes)
│   ├── dune_fd1_config.yaml          # DUNE Far Detector
│   ├── cubic_pixel_config.yaml       # Pixel readout test config
│   ├── production_*.yaml             # Profiler-tuned production configs (incl. doraemon)
│   ├── noise_spectrum.npz            # Empirical noise spectral shape
│   └── sce_jaxtpc.h5                 # Space charge effect correction maps
├── viewer/                       # Interactive 3D/2D HTML viewer + GIF/MP4 export
├── scripts/                      # Standalone utilities (e.g. wire-geometry export)
├── slurm/                        # SLURM array drivers for batch production
├── closure/                      # Differentiable reconstruction / closure studies
└── run_simulation.ipynb          # Interactive single-event simulation notebook
```

## Installation

### Dependencies

- JAX (with GPU support recommended)
- NumPy, SciPy
- Matplotlib
- H5py
- **hdf5plugin** — required to *read* production output (default `blosc-zstd` codec)
- PyYAML
- Pillow (viewer GIF/MP4 export)

```bash
pip install -r requirements.txt
```

For GPU support, install JAX per the [JAX installation guide](https://docs.jax.dev/en/latest/installation.html) (e.g. `pip install -U "jax[cuda12]"`).

**Note:** Use `python3` (not `python`).

### Notebook output stripping (one-time, per clone)

Notebook outputs are kept out of git via an [`nbstripout`](https://github.com/kynan/nbstripout) clean filter (`.gitattributes` is committed). The filter itself lives in each clone's `.git/config`, so after cloning run once:

```bash
pip install nbstripout
nbstripout --install --attributes .gitattributes
```

Committed/pushed copies of `*.ipynb` are then automatically stripped of outputs and execution counts, while your local working copies keep their outputs. Without this step your notebook commits will *not* be stripped.

## Quick Start

### Interactive notebook

```bash
jupyter notebook run_simulation.ipynb
```

### Python API

```python
from tools.simulation import DetectorSimulator
from tools.geometry import generate_detector
from tools.loader import load_event
from tools.config import create_track_hits_config
import jax

# Load configuration and create simulator
detector_config = generate_detector('config/cubic_wireplane_config.yaml')
simulator = DetectorSimulator(
    detector_config,
    include_track_hits=True,
    include_digitize=True,
)

# Load event (deposits are automatically transformed to local coordinates)
deposits = load_event('data.h5', simulator.config, event_idx=0)

# Run simulation
response_signals, track_hits_raw, deposits = simulator.process_event(
    deposits, key=jax.random.PRNGKey(42))

# Convert to sparse format
sparse = simulator.to_sparse(response_signals, threshold_enc=1200)

# Finalize track labels
track_hits = simulator.finalize_track_hits(track_hits_raw)
```

### Production batch

```bash
python3 production/run_batch.py --data events.h5 --events 1000 --bucketed --workers 2
python3 production/run_batch.py --data events.h5 --config config/dune_ndlar_config.yaml \
    --total-pad 70000 --response-chunk 10000 --bucketed
```

See `production/README.md` for pipeline details, CLI flags, and output schema.

### Differentiable path

```python
simulator = DetectorSimulator(detector_config, differentiable=True, n_segments=1000)
signals = simulator.forward_segments(params, positions_mm, de, dx=5.0)
# Gradients flow through velocity, lifetime, diffusion, recombination
```

## Features

- **GPU-accelerated**: Full JAX JIT compilation, `lax.scan` volume iteration
- **N-volume architecture**: Arbitrary number of detector volumes (2 to 70+ tested)
- **Wire and pixel readout**: Configurable per detector config
- **Local coordinates**: Deposits transformed to volume-local frame in loader; all volumes geometrically identical for physics
- **Scan/vmap iteration**: Volumes processed via `lax.scan` (default) or `jax.vmap`; one compiled body for any N
- **Electron drift**: Diffusion (spatial convolution kernel generation) and lifetime attenuation
- **Angle-dependent recombination**: Modified Box (ArgoNeuT) and EMB (ICARUS 2024) models
- **Electronics response**: RC-RC convolution via sparse FFT
- **Intrinsic noise**: Wire-length-dependent noise model (MicroBooNE)
- **ADC digitization**: Configurable bit depth, pedestal, gain
- **Space charge effects**: Per-volume SCE maps loaded in local frame
- **Track correspondence**: Group-based 3D-to-2D mapping with Q_s disaggregation fractions
- **Differentiable path**: `jax.remat` + scan for gradients through all physics parameters
- **Threaded production**: Overlapped GPU simulation with CPU save workers
- **Production profiler**: Auto-tune total_pad, chunk sizes, max_keys from data

## Architecture

### Local Coordinates

The loader transforms deposits to volume-local coordinates:
```
x_local = drift_direction * (x_anode - x_global)    # distance from anode, >= 0
y_local = y_global - y_center
z_local = z_global - z_center
```

In local frame, all volumes share reference geometry (anode at x=0, drift toward -x, yz centered). The physics uses fixed constants — no per-volume geometry indexing needed in the scan body. Seg files save global positions (inverse transform applied before writing).

### Volume Iteration

All volumes are processed by a single `lax.scan` (or `vmap`) body compiled once. The body handles recombination, drift, per-plane wire response, electronics, noise, digitization, and track labeling. Plane loops (typically 3 for wire) are unrolled at trace time inside the body.

### Configuration

- **`SimConfig`** (static, closure-captured) — Array dimensions, mode flags, volume geometry. Changing triggers recompilation.
- **`SimParams`** (dynamic, JIT argument) — Physics scalars (velocity, lifetime, diffusion, recombination). Changeable per-call.

## Input Data Format

HDF5 files with particle segments from simulation (e.g., Geant4):
- `position`: (N, 3) — x, y, z in mm
- `dE`: (N,) — energy deposits in MeV
- `dx`: (N,) — step length in mm
- `theta`: (N,) — polar angle
- `phi`: (N,) — azimuthal angle
- `track_id`: (N,) — particle track IDs

## Output

The simulator returns `(response_signals, track_hits_raw, deposits)`:

1. **response_signals**: `{(vol_idx, plane_idx): array}` — wire signals (dense, bucketed, or wire-sparse)
2. **track_hits_raw**: `{(vol_idx, plane_idx): tuple}` — raw group correspondence for track labeling
3. **deposits**: `DepositData` — input deposits with `charge`, `photons`, `qs_fractions` filled

## Detector Configurations

| Config | Volumes | Readout | Description |
|--------|---------|---------|-------------|
| `cubic_wireplane_config.yaml` | 2 | Wire (U/V/Y) | Default, SBND-scale |
| `sbnd_config.yaml` | 2 | Wire | SBND |
| `microboone_config.yaml` | 1 | Wire | MicroBooNE |
| `icarus_config.yaml` | 4 | Wire | ICARUS |
| `dune_ndlar_config.yaml` | 70 | Wire | DUNE ND-LAr (5x7 module grid) |
| `dune_fd1_config.yaml` | 2 | Wire | DUNE Far Detector |
| `cubic_pixel_config.yaml` | 2 | Pixel | Pixel readout (cubic geometry) |

Profiler-tuned `production_*.yaml` variants (including the doraemon wire/pixel configs) are also under `config/`.

## Simulation Parameters

| Parameter | Default | Description |
|---|---|---|
| `total_pad` | 200,000 | Padded array size per volume (sets JIT shape) |
| `response_chunk_size` | 50,000 | Deposits per fori_loop iteration |
| `iterate_mode` | `'scan'` | Volume iteration: `'scan'` or `'vmap'` |
| `use_bucketed` | False | Optional wire-only sparse bucket accumulation (memory saver; not required for pixel) |
| `max_active_buckets` | 1,000 | Max buckets per plane (bucketed mode) |
| `include_noise` | False | Enable intrinsic noise |
| `include_electronics` | False | Enable RC-RC electronics response |
| `include_digitize` | False | Enable ADC digitization |
| `include_track_hits` | True | Enable track correspondence |
| `include_electric_dist` | False | Enable space charge effects |
| `differentiable` | False | Enable differentiable path (with `n_segments`) |
