# Gradients & Differentiable Simulation

JAXTPC's forward is fully differentiable (`DetectorSimulator(differentiable=True)`,
`forward_segments`, `jax.remat`). These notebooks teach that machinery — the same
one the [closure / reconstruction methods](../reco) are built on. The signature
loss is the **Sobolev** loss (`tools/losses.py`): convolving signals with the
Sobolev kernel `1/(k²+κ²)^{s/2}` gives non-overlapping signals a usable gradient.

| Notebook | What it shows | Verified |
|---|---|---|
| `forward_segments.ipynb` | the differentiable forward for point-charge segments; AD-vs-finite-difference agreement (in float64) | gradients ✓ |
| `gradient_visualizations.ipynb` | `jax.value_and_grad` of the Sobolev loss vs a target event; visualize ∂loss/∂`[x,y,z,dE]` and the signal residual | ✓ |
| `optimization.ipynb` | a **minimal closure**: Adam on `(N,4)` point charges, Sobolev loss falling (e.g. 1.04 → 0.56 over 40 steps) | ✓ end-to-end |
| `particle_generation.ipynb` | differentiable muon tracks (`tools/particle_generator.py`); gradients wrt track start/direction | — |

`optimization.ipynb` is deliberately the bridge to `reco/`: it is the segments
closure stripped to its core (forward → Sobolev loss → `value_and_grad` → Adam),
so the reco notebooks can focus on the full method (MCMC relocation, track
parameterizations, real-scale events).

> Run gradient-correctness comparisons in **float64** — comparing an AD gradient
> to a finite difference of a float32 loss is numerically fragile.
