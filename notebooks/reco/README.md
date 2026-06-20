# Reconstruction

Reconstruction demos and studies built on the (differentiable) simulator —
recovering deposit/track quantities from the simulated readout.

**Planned**
- `gradient_reco.ipynb` — gradient-based reconstruction: optimize deposit
  positions/charge to match an observed event through the differentiable forward
- `learned_reco.ipynb` — feed-forward NN reconstruction (e.g. epipolar
  y-recovery + charge density) from wire projections
- `single_particle_closure.ipynb` — single-particle closure study: how well are
  charge / dE / geometry recovered, and where are the failure modes
- `space_points.ipynb` — rough 3D reconstruction from wire crossings
  (`tools/space_points.py`)

These depend on which reconstruction approaches we want to showcase; this folder
is the home for reco tutorials and reproducible studies.
