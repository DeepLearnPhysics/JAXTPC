# ElectricDistortion

Space charge effect (SCE) simulation for liquid argon TPCs. Computes the steady-state
ion charge density, solves the Poisson equation for the distortion potential, calculates
the full 3D electric field, and ray-traces electron drift paths to produce spatial
distortion maps.

## Quick Start

### Generate SCE maps for a detector

```bash
# JAXTPC detector (default: 101^3 Poisson grid, 26^3 output grid)
python -m ElectricDistortion --detector jaxtpc --output sce_maps_jaxtpc.npz

# Quick test with coarse grids
python -m ElectricDistortion --detector jaxtpc --quick --output sce_maps_quick.npz

# Custom parameters
python -m ElectricDistortion --detector sbnd --E0 500 --Nxo 41 --Nyo 41 --Nzo 41

# Self-consistent ion drift velocity iteration
python -m ElectricDistortion --detector jaxtpc --self-consistent --output sce_maps_sc.npz
```

### Load maps in Python

```python
from ElectricDistortion.io.map_io import load_maps_npz

maps = load_maps_npz('sce_maps_jaxtpc.npz')
delta_x = maps['delta_x']   # (Nx_out, Ny_out, Nz_out) spatial distortion in cm
delta_y = maps['delta_y']
delta_z = maps['delta_z']
Ex = maps['Ex']              # (Nx_poisson, Ny_poisson, Nz_poisson) E-field in V/cm
Ey, Ez = maps['Ey'], maps['Ez']
params = maps['params']      # dict with Lx, Ly, Lz, E0, etc.
```

### Use the drift velocity function

```python
from ElectricDistortion.core.drift_velocity import drift_velocity

v = drift_velocity(500.0, T=89.0)  # cm/us at 500 V/cm, 89 K
```

## Physics Pipeline

1. **Charge density** (`core/physics.py`): Steady-state ion density
   $\rho(x) = Q \cdot x / v_\mathrm{ion}$, uniform in y/z.

2. **Poisson solve** (`core/physics.py`): 3D DST-based solver for the distortion
   potential $\delta\phi$ with homogeneous Dirichlet BCs on all faces.

3. **E-field** (`core/physics.py`): Total physical field
   $E_x = E_0 - \partial(\delta\phi)/\partial x$, $E_y = -\partial(\delta\phi)/\partial y$,
   $E_z = -\partial(\delta\phi)/\partial z$.

4. **Electron tracing** (`core/electron_drift.py`): ODE integration (RK45) of
   individual electron trajectories through the distorted field to the anode.
   Parallelised via `multiprocessing.Pool`.

5. **Distortion maps**: $\Delta_x = v_0 \cdot t_\mathrm{drift} - x_0$ (drift time
   distortion), $\Delta_y = y_\mathrm{anode} - y_0$, $\Delta_z = z_\mathrm{anode} - z_0$.

6. **Drift velocity** (`core/drift_velocity.py`): Walkowiak parameterisation
   (NIM A 449, 2000) with LArSoft/ICARUS parameters.

## Package Structure

```
ElectricDistortion/
  core/
    drift_velocity.py    # Walkowiak v(E) parameterisation
    electron_drift.py    # ODE-based electron ray tracing
    physics.py           # Charge density, Poisson solver, E-field
  io/
    config_loader.py     # YAML config + detector presets
    map_io.py            # Save/load distortion maps (npz, hdf5)
  config/
    sce_config.yaml      # Default simulation parameters
    detector_presets.yaml # MicroBooNE, SBND, ICARUS, DUNE, JAXTPC
  plotting/
    visualize.py         # Basic slice/profile plots
    advanced.py          # Warped grids, streamlines, quiver plots
  run_sce.py             # Main simulation driver + CLI
  validate_sce.py        # Physics validation suite
  expert_review.py       # Detailed physics checks (drift velocity, Gauss's law, etc.)
  generate_plots.py      # Batch plot generation
  PHYSICS.md             # Detailed physics documentation
```

## Available Detector Presets

| Preset | Drift (cm) | Transverse (cm) | E0 (V/cm) |
|--------|-----------|-----------------|-----------|
| `microboone` | 256 | 233 x 1037 | 273 |
| `sbnd` | 200 | 400 x 500 | 500 |
| `icarus` | 150 | 360 x 1960 | 500 |
| `jaxtpc` | 216 | 432 x 432 | 500 |
| `dune_fd_hd` | 363 | 600 x 1200 | 500 |
| `dune_fd_vd` | 650 | 600 x 1200 | 500 |

## Dependencies

- numpy
- scipy (Poisson solver, ODE integration, interpolation)
- matplotlib (plotting, optional for map generation)
- pyyaml (config loading)
- h5py (optional, for HDF5 output)
