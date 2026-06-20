# Gradients & Differentiable Simulation

JAXTPC has a fully differentiable forward path (`forward` / `forward_segments`,
`jax.remat` for memory-efficient reverse-mode gradients through velocity,
lifetime, diffusion, and recombination parameters). These notebooks visualize
and exploit those gradients.

**Planned**
- `gradient_visualizations.ipynb` — visualize ∂(signal)/∂(physics params); how a
  signal responds to velocity / lifetime / diffusion / recombination; gradient
  fields and loss landscapes
- `forward_segments.ipynb` — the lightweight differentiable forward for
  segment-like data; finite-difference vs autodiff agreement
- `optimization.ipynb` — fit physics parameters by gradient descent on a target
  event; the multi-scale spectral and point-cloud (OT) losses (`tools/losses.py`,
  `tools/pointcloud.py`)
- `particle_generation.ipynb` — differentiable muon track generation
  (`tools/particle_generator.py`) for end-to-end optimization

See `docs/differentiable/` for the written companion pages.

> Differentiable path requires `differentiable=True, n_segments=N` at simulator
> construction. Gradient-correctness checks should run in float64 (the
> finite-difference comparison is float32-fragile otherwise).
