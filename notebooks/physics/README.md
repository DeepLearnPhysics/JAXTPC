# Physics

The detector response chain, stage by stage. For understanding *what the
simulation is doing* and exploring the physics models.

| Notebook | What it does |
|---|---|
| `response_kernels.ipynb` | Visualize the wire response kernels and the diffusion-convolved `DKernel` tables; interpolation sweeps |
| `physics_walkthrough.ipynb` | One synthetic track with a plot at each pipeline stage: deposits → recombination (Q/L) → drift/diffusion/response → wire signals |

**Planned**
- `recombination_models.ipynb` — modified-box vs EMB vs passthrough; angular dependence
- `drift_diffusion.ipynb` — drift velocity, lifetime, longitudinal/transverse diffusion
- `space_charge.ipynb` — SCE: generate/load E-field maps, apply the (time-primary) corrections, visualize distortion

See `docs/physics/` for the written companion pages.
