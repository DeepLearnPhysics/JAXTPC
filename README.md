# JAXTPC

## Overview

JAXTPC is a GPU-accelerated Time Projection Chamber (TPC) simulation framework built with JAX. It models particle interactions in liquid argon TPCs — drift, diffusion, recombination, signal formation, electronics response, and noise.

## Repository Structure

```
JAXTPC/
├── tools/                        # Core simulation package
│   ├── simulation.py             # DetectorSimulator class
│   ├── config.py                 # Data types (SimParams, SimConfig, DepositData, ...)
│   ├── physics.py                # Shared physics pipeline (side/plane computations)
│   ├── geometry.py               # Detector configuration parser from YAML
│   ├── kernels.py                # DCT-based diffusion kernel generation + interpolation
│   ├── drift.py                  # Electron drift physics
│   ├── wires.py                  # Wire signal calculations with diffusion
│   ├── recombination.py          # Charge recombination (Modified Box + EMB models)
│   ├── electronics.py            # RC-RC electronics response
│   ├── noise.py                  # Intrinsic noise generation
│   ├── track_hits.py             # Group-based track labeling + Q_s fractions
│   ├── loader.py                 # HDF5 I/O, group assignment, east/west splitting
│   ├── output.py                 # Format conversion (dense, bucketed, sparse)
│   ├── visualization.py          # Wire signal / track label / diffused charge plotting
│   ├── particle_generator.py     # Muon track generation (numpy + JAX)
│   ├── losses.py                 # Sobolev / spectral loss functions
│   └── responses/                # Pre-computed wire response kernels (U/V/Y NPZ)
├── production/                   # Batch production pipeline
│   ├── run_batch.py              # Batch simulation with threaded save workers
│   ├── save.py                   # HDF5 save functions (resp/seg/corr)
│   ├── load.py                   # HDF5 load/decode functions
│   ├── view_production.ipynb     # Visualize production output (no simulation needed)
│   └── DATA_FORMAT.md            # Output file schema documentation
├── config/                       # Detector configuration
│   └── cubic_wireplane_config.yaml
└── run_simulation.ipynb          # Interactive simulation notebook
```

## Installation

### Dependencies

- JAX (with GPU support recommended)
- NumPy
- Matplotlib
- H5py
- PyYAML

```bash
pip install jax[cuda] numpy matplotlib h5py pyyaml
```

For GPU support, follow the [JAX installation guide](https://github.com/google/jax#installation).

**Note:** Use `python3` (not `python`).

## Quick Start

### Interactive notebook

```bash
jupyter notebook run_simulation.ipynb
```

### Python API

```python
from tools.simulation import DetectorSimulator
from tools.geometry import generate_detector
from tools.loader import load_particle_step_data

# Load configuration and event data
detector_config = generate_detector('config/cubic_wireplane_config.yaml')
deposit_data, group_to_track = load_particle_step_data('data.h5', event_idx=0)

# Create simulator
simulator = DetectorSimulator(
    detector_config,
    include_noise=False,
    include_electronics=False,
    include_track_hits=True,
)

# Run simulation
response_signals, track_hits, qs_fractions = simulator(deposit_data)

# Convert to dense arrays for visualization
from tools.output import to_dense
dense = to_dense(response_signals, simulator.config)
```

### Production batch

```bash
python3 production/run_batch.py --data events.h5 --events 1000 --workers 2
```

See `production/DATA_FORMAT.md` for output schema and batch pipeline details.

## Features

- **GPU-accelerated**: Full JAX JIT compilation for GPU
- **Dual-sided TPC**: Simulates both east and west drift regions
- **Three wire planes**: U, V, Y induction and collection planes per side
- **Electron drift**: Diffusion (DCT-based kernel generation) and lifetime attenuation
- **Recombination models**: Modified Box (ArgoNeuT) and EMB (ICARUS 2024) with angular dependence
- **Electronics response**: RC-RC convolution via sparse FFT
- **Intrinsic noise**: Wire-length-dependent noise model (MicroBooNE)
- **ADC digitization**: Configurable bit depth, pedestal, gain
- **Track correspondence**: Group-based 3D-to-2D mapping with Q_s disaggregation fractions
- **Threaded production**: Overlapped GPU simulation with CPU save workers

## Input Data Format

HDF5 files with particle segments from simulation (e.g., Geant4):
- `position`: (N, 3) — x, y, z in mm
- `dE`: (N,) — energy deposits in MeV
- `dx`: (N,) — step length in mm
- `theta`: (N,) — polar angle
- `phi`: (N,) — azimuthal angle
- `track_id`: (N,) — particle track IDs

## Output

The simulator returns `(response_signals, track_hits, qs_fractions)`:

1. **response_signals**: `{(side, plane): array}` — wire signals (dense, bucketed, or wire-sparse depending on config)
2. **track_hits**: `{(side, plane): tuple}` — raw group correspondence for track labeling
3. **qs_fractions**: `(N,)` float32 — each deposit's charge fraction within its group (None if track hits disabled)

## Configuration

Detector parameters in `config/cubic_wireplane_config.yaml`:
- Detector dimensions (dual TPC, 6 wire planes)
- Wire plane geometry (pitch, angles, spacing)
- Drift parameters (velocity, diffusion coefficients, electron lifetime)
- Recombination model (Modified Box or EMB with angular dependence)
- Readout (time discretization, ADC conversion, digitization)

### Simulation knobs

| Parameter | Default | Effect |
|---|---|---|
| `total_pad` | 200,000 | Padded array size per side (sets JIT shape) |
| `response_chunk_size` | 50,000 | Deposits per fori_loop iteration |
| `max_keys` | 4,000,000 | Max unique hits for track labeling |
| `include_noise` | False | Enable intrinsic noise |
| `include_electronics` | False | Enable RC-RC electronics response |
| `include_digitize` | False | Enable ADC digitization |
| `include_track_hits` | True | Enable track correspondence |
| `use_bucketed` | False | Sparse bucket accumulation mode |
