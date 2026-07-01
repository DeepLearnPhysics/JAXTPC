# Differentiable path

JAXTPC exposes a fully differentiable forward simulation so you can
back-propagate through the detector physics — for gradient-based
reconstruction, parameter fitting, and calibration. This page covers what the
differentiable path *is*; see [optimization](optimization.md) for a runnable
fitting loop.

The differentiable path is one of three [execution paths](../architecture/execution-paths.md).
It shares the exact same physics body (`compute_volume_physics` →
`compute_plane_physics` → `compute_plane_signal`) as production — only the
wrapper differs.

## Enabling it

Construct the simulator with `differentiable=True` and a fixed segment count:

```python
from tools.simulation import DetectorSimulator

sim = DetectorSimulator(detector_config, differentiable=True, n_segments=20_000)
```

`n_segments` replaces `total_pad`: all deposit arrays are padded to this fixed
length so the accumulation loop bound is a compile-time constant. This is what
makes reverse-mode differentiation through the `fori_loop` tractable.

!!! warning "Wire-only — pixel is unsupported"
    Constructing with `differentiable=True` on a pixel-readout detector raises
    `ValueError` (`tools/simulation.py:122`). The differentiable forward is
    built only when `self.n_segments is not None and self._readout_type == 'wire'`.
    The pixel differentiable-response factory is a **placeholder** aliased to the
    production one (`_build_response_fn_diff = _build_response_fn`,
    `tools/simulation.py:422`); it is never reached because construction fails
    first.

!!! note "Response only — no readout post-processing"
    The differentiable path is **response-only**. It produces the field response
    signal and nothing else: **no** intrinsic/coherent noise, **no** electronics
    shaping, **no** digitization, and **no** track-hit labeling. Those stages
    live only in `process_event`.

## Two entry points

Both call the same internal `_forward_diff` (a `jax.remat`-wrapped
`iterate` over volumes) and return a tuple of signal arrays, one per
`(volume, plane)`.

| Method | Input coordinates | Padding | Use when |
|---|---|---|---|
| `forward(params, deposits)` | **local** frame `DepositData` | pads to `total_pad` (= `n_segments`) | you already have split, local-frame deposits |
| `forward_segments(params, positions_mm, de, dx)` | **global** positions + `de`/`dx` | pads internally | you want a traceable call directly inside `jax.grad` |

`forward` expects a `DepositData` already in local coordinates and pads it to
`total_pad` (`tools/simulation.py:947`).

`forward_segments` takes raw **global** positions plus `de`/`dx`, does the
local-frame transform and per-volume masking *inside* the traced function
(`tools/simulation.py:970`). Volume selection uses `jax.lax.stop_gradient`
masks — no NumPy splitting — so the whole thing is traceable and can be nested
inside `jax.grad`. Volume `de` is zeroed for out-of-range positions:

```python
vol_mask  = jax.lax.stop_gradient((x_cm >= x_min) & (x_cm < x_max))
masked_de = padded_de * vol_mask
x_local   = vol.drift_direction * (vol.x_anode_cm * 10.0 - padded_pos[:, 0])
```

## What carries gradients

Gradients flow through the physics scalars carried in `SimParams`:

- `velocity_cm_us` — drift velocity
- `lifetime_us` — electron lifetime (attenuation)
- `diffusion_trans_cm2_us`, `diffusion_long_cm2_us` — diffusion coefficients
- `recomb_params` — recombination model parameters

Diffusion is differentiable because the differentiable response factory
(`_build_response_fn_diff`) regenerates the `DKernel` diffusion table from the
current `SimParams` diffusion values on every call, with **static** Gaussian
filter sizes (`ks_w`/`ks_t`, precomputed from the default params). Only the
Gaussian *widths* change with the parameters, not the graph shape — so the trace
stays fixed while the kernel stays differentiable.

`forward_segments` also carries gradients through `positions_mm`, `de`, and
`dx`, which is what enables track-geometry fitting (see
[particle generation](particle-generation.md)).

## `remat` for memory

The per-volume forward and `_forward_diff` are both wrapped in `jax.remat`
(`tools/simulation.py:669`, `:680`). Reverse-mode gradients through the deep
response graph would otherwise need to store every intermediate activation;
`remat` recomputes them on the backward pass instead, trading compute for a much
smaller memory footprint. This is essential for the response accumulation, which
touches large `(num_wires, num_time)` arrays.

## Contrast with `process_event`

| | `forward` / `forward_segments` | `process_event` |
|---|---|---|
| Differentiable | **yes** | no |
| Loop bound `n_actual` | static `int` (`n_segments`) | traced JAX value |
| Wrapper | `jax.remat` | `jax.lax.fori_loop`, plain JIT |
| Readout | **wire only** | wire **and** pixel |
| Post-processing | none (response only) | electronics, noise, digitize, track-hits |
| Returns | tuple of signal arrays | `(response_signals, track_hits, deposits)` |

The single structural divergence is the loop-bound computation inside
`compute_plane_signal`, gated on `isinstance(n_actual, int)` — a Python `int`
(differentiable) selects a static `min(...)` bound, a traced value (production)
selects `jnp.minimum(...)`. See
[execution paths](../architecture/execution-paths.md) for the full comparison.

## Charge + photons only: `process_event_light`

If you only need recombination output (ionization charge `Q` and scintillation
photons `L`) and drift-derived quantities — not the readout signal — use
`process_event_light`. It runs `compute_volume_physics` and returns a
`DepositData` with `charge` and `photons` filled, skipping the entire
response/accumulation loop:

```python
deposits = sim.process_event_light(deposits)
deposits.volumes[0].charge   # recombined ionization electrons per segment
deposits.volumes[0].photons  # scintillation photons per segment
```

This is not the differentiable path (it is a plain JIT call), but it is the
cheapest way to get `Q`/`L` when the readout signal is not needed.

## A minimal gradient example

`forward_segments` is the traceable entry point, so `jax.grad` composes with it
directly. Here we take the gradient of a scalar loss with respect to the drift
velocity:

```python
import jax
import jax.numpy as jnp

sim = DetectorSimulator(detector_config, differentiable=True, n_segments=20_000)
params = sim.default_sim_params

# a straight segment of deposits in GLOBAL mm coordinates
positions_mm = jnp.stack([
    jnp.linspace(-100.0, -50.0, 200),   # x (drifts)
    jnp.zeros(200),                      # y
    jnp.linspace(0.0, 30.0, 200),        # z
], axis=1)
de = jnp.full(200, 2.1)   # MeV per step
dx = 0.3                  # cm

def loss(p):
    signals = sim.forward_segments(p, positions_mm, de, dx)
    return sum(jnp.sum(s**2) for s in signals)   # scalar

g = jax.grad(loss)(params)
print(g.velocity_cm_us, g.lifetime_us)   # d loss / d physics params
```

`jax.grad(loss)(params)` returns a `SimParams` pytree of gradients — one entry
per differentiable field. To fit *tracks* instead of detector parameters,
differentiate with respect to `positions_mm`/`de` (or a track parameterization);
see [optimization](optimization.md) and
[particle generation](particle-generation.md).

## See also

- [Execution paths](../architecture/execution-paths.md) — the three entry points
  and the shared physics body.
- [Optimization](optimization.md) — end-to-end gradient fitting loop.
- [Losses](losses.md) — differentiable signal-comparison losses.
- [Particle generation](particle-generation.md) — differentiable muon tracks.
