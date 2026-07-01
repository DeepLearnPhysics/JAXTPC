# Optimization

The [differentiable path](overview.md) turns reconstruction and calibration into
plain gradient descent: define a scalar loss between a simulated signal and a
target, take its gradient with respect to whatever you want to fit (physics
parameters or track geometry), and step an optimizer. This page is a practical
how-to for that loop.

!!! tip "Runnable notebook"
    A complete, CI-runnable example lives in
    [`notebooks/gradients/optimization.ipynb`](https://github.com/DeepLearnPhysics/JAXTPC/blob/main/notebooks/gradients/optimization.ipynb).
    Start there; this page is the annotated skeleton behind it.

## The pattern

Every fit is the same four-line loop:

1. **Forward** — simulate a signal from the current parameters with
   `forward_segments` (or `forward`).
2. **Loss** — compare it to the target signal with a differentiable loss
   (see [losses](losses.md)).
3. **Grad** — `jax.grad` (or `jax.value_and_grad`) of the loss.
4. **Update** — step an optimizer and repeat.

Because `forward_segments` is fully traceable, the loss composes directly with
`jax.grad`, and the whole step JIT-compiles.

## What you can fit

| Target of the gradient | You are fitting | Differentiate w.r.t. |
|---|---|---|
| `SimParams` fields | detector physics (velocity, lifetime, diffusion, recombination) | `params` |
| `positions_mm` / `de` / a track parameterization | track geometry / energy | the track args |

Both are gradient descent; they differ only in which argument of the loss you
differentiate. See [particle generation](particle-generation.md) for a
differentiable muon track parameterization (`build_muon_forward`) you can fit
end-to-end with a handful of angle/energy scalars instead of raw per-step arrays.

## Setup

Build the simulator once in differentiable mode; reuse it every step (the JIT
compiles on the first call and is cached afterward):

```python
import jax
import jax.numpy as jnp
from tools.simulation import DetectorSimulator

sim = DetectorSimulator(detector_config, differentiable=True, n_segments=20_000)

# Target: a "measured" signal to reconstruct against.
target_signals = sim.forward_segments(true_params, true_pos, true_de, dx)
```

## Fitting physics parameters

Optimize `SimParams` so the simulated signal matches the target. This uses an
`optax`-style loop (any optimizer works; Adam is a good default):

```python
import optax
from tools.losses import blur_mse_loss, make_spectral_weight, DEFAULT_BLUR_SIGMAS

# Precompute one spectral weight per plane shape (host-side, before JIT).
spectral_weights = tuple(
    make_spectral_weight(s.shape[0], s.shape[1], DEFAULT_BLUR_SIGMAS)
    for s in target_signals)

def loss_fn(params):
    signals = sim.forward_segments(params, positions_mm, de, dx)
    return blur_mse_loss(signals, target_signals, spectral_weights)

opt = optax.adam(1e-2)
params = sim.default_sim_params
opt_state = opt.init(params)

@jax.jit
def step(params, opt_state):
    loss, grads = jax.value_and_grad(loss_fn)(params)
    updates, opt_state = opt.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss

for i in range(500):
    params, opt_state, loss = step(params, opt_state)
    if i % 50 == 0:
        print(i, float(loss))
```

`grads` is a `SimParams` pytree with one entry per differentiable field
(`velocity_cm_us`, `lifetime_us`, `diffusion_*`, `recomb_params`); `optax`
updates the whole pytree at once. See
[what carries gradients](overview.md#what-carries-gradients).

## Fitting track geometry

To reconstruct a track, hold the physics fixed and differentiate the loss with
respect to the track arguments instead:

```python
def loss_fn(track_pos):
    signals = sim.forward_segments(params, track_pos, de, dx)
    return blur_mse_loss(signals, target_signals, spectral_weights)

opt = optax.adam(1e-1)
track_pos = initial_guess_mm            # (N, 3) global coordinates
opt_state = opt.init(track_pos)

@jax.jit
def step(track_pos, opt_state):
    loss, grads = jax.value_and_grad(loss_fn)(track_pos)
    updates, opt_state = opt.update(grads, opt_state)
    return optax.apply_updates(track_pos, updates), opt_state, loss

for i in range(300):
    track_pos, opt_state, loss = step(track_pos, opt_state)
```

For a physically-constrained fit (far fewer free parameters, better
conditioning), parameterize the track with
[particle generation](particle-generation.md) and differentiate w.r.t. the
generator's angle/energy/start scalars rather than raw positions.

## Practical notes

- **Loss choice matters.** A raw per-pixel MSE has almost no gradient when
  tracks barely overlap. `blur_mse_loss` combines many blur scales
  (`DEFAULT_BLUR_SIGMAS`) into one precomputed spectral weight so gradients
  reach across mis-aligned signals — see [losses](losses.md) for the full menu
  (Parseval blur MSE, Sobolev, geomean/log1p plane rebalancing).
- **`value_and_grad`** gives you the loss for free alongside the gradient — cheaper
  than a separate forward for logging.
- **JIT the step, not the loop.** Wrap the whole `step` in `@jax.jit` so the
  forward, loss, grad, and optimizer update fuse into one compiled call.
- **Memory** comes from `remat` in the forward (see
  [overview](overview.md#remat-for-memory)); if you still OOM, lower
  `n_segments` or `response_chunk_size`.
- **Wire only.** The differentiable path does not support pixel readout — see the
  [overview warning](overview.md#enabling-it).

## See also

- [Differentiable overview](overview.md) — how `forward` / `forward_segments`
  and gradients work.
- [Losses](losses.md) — the differentiable comparison losses.
- [Particle generation](particle-generation.md) — differentiable track
  parameterization for geometry fits.
- [`optimization.ipynb`](https://github.com/DeepLearnPhysics/JAXTPC/blob/main/notebooks/gradients/optimization.ipynb)
  — the full runnable example.
