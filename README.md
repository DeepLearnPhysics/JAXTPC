# JAXTPC

## Overview

JAXTPC is a high-performance Time Projection Chamber (TPC) simulation framework built with JAX. The system models particle interactions in liquid argon TPCs, including drift, diffusion, recombination, and signal formation on wire planes.

## Repository Structure

```
JAXTPC/
├── tools/                    # Core simulation modules
│   ├── config.py             # Data structures and configuration classes
│   ├── drift.py              # Electron drift physics calculations
│   ├── geometry.py           # Detector geometry parser
│   ├── kernels.py            # Diffusion kernel generation and application
│   ├── loader.py             # HDF5 particle step data I/O
│   ├── recombination.py      # Charge recombination models (Box model)
│   ├── simulation.py         # Main DetectorSimulator class
│   ├── sparse_utils.py       # Sparse format conversion utilities
│   ├── track_hits.py         # Track-level hit extraction
│   ├── visualization.py      # Wire signal plotting
│   ├── wires.py              # Wire signal calculations
│   └── responses/            # Pre-computed wire response kernels
│       ├── U_plane_kernel.npz
│       ├── V_plane_kernel.npz
│       └── Y_plane_kernel.npz
├── config/                   # Detector configuration files
│   └── cubic_wireplane_config.yaml
├── run_simulation_clean.ipynb  # Example simulation notebook
└── muon.h5                   # Example particle data
```

## Installation

### Dependencies

- JAX (with GPU support recommended)
- NumPy
- Matplotlib
- H5py
- PyYAML
- SciPy

Install with:
```bash
pip install jax jaxlib numpy matplotlib h5py pyyaml scipy
```

For GPU support, follow the [JAX installation guide](https://github.com/google/jax#installation).

## Quick Start

Run the example notebook:

```bash
jupyter notebook run_simulation_clean.ipynb
```

Or use the Python API directly:

```python
from tools.simulation import DetectorSimulator
from tools.config import DepositData, create_diffusion_params, create_track_hits_config
from tools.geometry import generate_detector
from tools.loader import load_particle_step_data
from tools.recombination import recombine_steps
import jax.numpy as jnp

# Load configuration and data
detector_config = generate_detector('config/cubic_wireplane_config.yaml')
step_data = load_particle_step_data('muon.h5', event_idx=0)

# Create diffusion parameters
diffusion_params = create_diffusion_params(
    max_sigma_trans_unitless=detector_config['max_sigma_trans_unitless'],
    max_sigma_long_unitless=detector_config['max_sigma_long_unitless'],
    num_s=16
)

# Prepare input data (N segments from particle simulation)
positions_mm = jnp.asarray(step_data['position'], dtype=jnp.float32)
charges = jnp.asarray(recombine_steps(step_data, detector_config), dtype=jnp.float32)
n_segments = positions_mm.shape[0]

deposit_data = DepositData(
    positions_mm=positions_mm,
    charges=charges,
    valid_mask=jnp.ones(n_segments, dtype=bool),
    theta=jnp.asarray(step_data.get('theta', jnp.zeros(n_segments)), dtype=jnp.float32),
    phi=jnp.asarray(step_data.get('phi', jnp.zeros(n_segments)), dtype=jnp.float32),
    track_ids=jnp.asarray(step_data.get('track_id', jnp.ones(n_segments)), dtype=jnp.int32)
)

# Create simulator and run
simulator = DetectorSimulator(
    detector_config,
    response_path="tools/responses/",
    diffusion_params=diffusion_params,
    use_bucketed=True  # Memory-efficient sparse accumulation
)

response_signals, track_hits = simulator(deposit_data)
```

## Features

- **GPU-accelerated**: Full JAX JIT compilation for GPU/TPU
- **Dual-sided TPC**: Simulates both east and west drift regions
- **Three wire planes**: U, V, Y induction and collection planes
- **Electron drift**: Includes diffusion and lifetime attenuation
- **Angle-dependent response**: Wire signals depend on track angle
- **Box model recombination**: Converts energy deposits to electron counts
- **Sparse output**: Memory-efficient bucketed accumulation
- **Truth labeling**: Preserves particle track IDs for truth-matching

## Input Data Format

HDF5 files containing N particle segments from simulation (e.g., Geant4):
- `position`: (N, 3) array of x, y, z positions in mm
- `dE`: (N,) array of energy deposits in MeV
- `theta`: (N,) array of polar angles
- `phi`: (N,) array of azimuthal angles
- `track_id`: (N,) array of particle track IDs

## Output Format

The simulator returns two outputs:

1. **track_hits**: Truth-level hits with track ID labels preserved
   - Dictionary mapping `(side_idx, plane_idx)` to hit data
   - Each hit contains: wire index, time index, charge, track_id
   - Used for truth-matching and training labels

2. **response_signals**: Detector response (track_hits convolved with wire response)
   - Dictionary mapping `(side_idx, plane_idx)` to wire signals
   - In bucketed mode: `(buckets, num_active, compact_to_key, B1, B2)`
   - In dense mode: 2D arrays of shape `(num_wires, num_time_steps)`
   - Represents realistic detector output

## Configuration

Detector parameters are defined in `config/cubic_wireplane_config.yaml`:
- Detector dimensions
- Wire plane geometry (pitch, angles)
- Drift parameters (velocity, diffusion coefficients, lifetime)
- Recombination model parameters
- Time discretization settings
