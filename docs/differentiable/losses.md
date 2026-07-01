# Losses

Optimizing detector parameters or a particle track against a target readout
needs a loss that produces **useful gradients on sparse signals**. A liquid-argon
wireplane image is mostly empty: a muon lights up a thin curve through a
`(num_wires, num_time)` grid that is 99%+ zeros. Plain pixelwise MSE fails here —
if the simulated track and the target track do not *overlap*, the per-pixel
difference is locally flat and the gradient carries no information about *which
way* to move the track. `tools/losses.py` fixes this by comparing signals in the
**frequency domain** with a weight that couples nearby pixels, so gradient
information flows across the empty space between two non-overlapping tracks.

All losses in this module share the same shape contract as the differentiable
path (see [overview](overview.md)): they take two tuples of per-plane 2-D arrays
(`signals_a`, `signals_b`) — the simulated and target signals, one array per
`(volume, plane)` pair — plus a tuple of precomputed spectral weights, and return
a scalar. Signal values are in [ENC (wire)](../physics/units.md), but every loss
normalizes by `sum(|target|)` internally, so the scale cancels and the loss is
dimensionless.

!!! tip "Which one should I use?"
    Start with **`make_sobolev_weight` + `sobolev_loss_geomean_log1p`**. It is the
    parameter-free, plane-balanced default that works across detectors. The other
    functions (`blur_mse_loss`, `sobolev_loss`, `sobolev_loss_geomean`) are
    research variants documented below for context and ablations.

## The common structure: Parseval, one FFT per plane

Every `*_loss_single` in this module is the same three lines of math. To compare
two signals at *many* blur scales at once, you would naively blur both signals
with each of `S` Gaussian kernels and sum the MSEs — `S` convolutions per plane.
Instead JAXTPC uses **Parseval's theorem** to collapse all scales into a single
frequency-domain weight `W(f)` computed once, then evaluates

```
loss = (1/N) * sum_f |FFT(A - B)(f)|^2 * W(f)
```

with a single `fft2` per plane. This is the whole trick: a weighted sum of the
difference's power spectrum, `O(N log N)`, no matter how many scales `W` encodes.
`W(f)` is precomputed **once per plane shape** (outside JIT) and reused across
every forward/backward pass.

```python
# tools/losses.py:blur_mse_loss_single (sobolev_loss_single is identical)
norm = jnp.sum(jnp.abs(B)) + 1e-12
diff = (A - B) / norm
diff_pad = jnp.pad(diff, ((pad_h, pad_h), (pad_w, pad_w)))   # zero-pad
diff_fft = jnp.fft.fft2(diff_pad)
power = diff_fft.real ** 2 + diff_fft.imag ** 2
return jnp.sum(power * spectral_weight) / N
```

The zero-padding (inferred at trace time from the weight's shape) turns the
periodic FFT convolution into a linear one, so a bright pixel near the image edge
does not wrap its gradient around to the far side.

## Multi-scale spectral blur MSE (`blur_mse_loss`)

The original loss. `make_spectral_weight(H, W, sigmas)` builds

```
W(f) = sum_s  sigma_s^2 * |G_hat_s(f)|^2
     = sum_s  sigma_s^2 * exp(-4 pi^2 sigma_s^2 |f|^2)
```

a **sum of Gaussians in frequency space**, one term per blur scale in `sigmas`.
The default ladder spans pixel-exact to detector-scale:

```python
DEFAULT_BLUR_SIGMAS = (0, 1, 2, 4, 8, 16, 32, 64, 128, 256)
```

`sigma = 0` is a delta kernel: `|G_hat|^2 = 1`, contributing a flat weight of `1`
— i.e. raw MSE recovered via Parseval. The large sigmas add low-frequency weight
so that two tracks separated by hundreds of wires still produce a gradient that
pulls them together. Reach is `~4*sigma` pixels; at 256 that is ~1024 bins.

```python
from tools.losses import blur_mse_loss, make_spectral_weight, DEFAULT_BLUR_SIGMAS

# Precompute one weight per unique plane shape (once, before JIT)
sw = tuple(make_spectral_weight(H, W, DEFAULT_BLUR_SIGMAS)
           for H, W in plane_shapes)

def loss_fn(params):
    sigs = forward(...)                  # tuple of per-plane arrays
    return blur_mse_loss(sigs, target_signals, sw)

loss, grads = jax.jit(jax.value_and_grad(loss_fn))(init_params)
```

The discrete sum of Gaussians makes the gradient magnitude-vs-distance profile
lumpy (each scale contributes a bump). The Sobolev weight below replaces it with
a single smooth power law and cleaner theory.

## Sobolev H⁻ˢ losses (`make_sobolev_weight`, `sobolev_loss`)

`make_sobolev_weight(H, W, max_pad, s)` swaps the sum-of-Gaussians for the
screened **negative-order Sobolev norm**

```
W(f) = 1 / (|f|^2 + eps)^s ,   eps = 1 / (pi^2 * max_pad^2)
```

This is the (regularized) `H^{-s}` norm, which for small perturbations is
equivalent to a **Wasserstein / optimal-transport distance** between the two
signals (Peyré 2018) — exactly the "how far, and which way, do I move charge"
quantity you want when fitting a track. The `1/|f|^2` shape **downweights high
frequencies**, so the loss cares about coarse spatial displacement first and
pixel-exact detail last, and the resulting gradient reaches far across empty
space.

The exponent `s` tunes the gradient's distance profile:

| `s`   | Loss growth vs displacement `d` | Gradient | Analogue |
|-------|--------------------------------|----------|----------|
| `1`   | `log(d)`                       | `1/d`    | Laplacian / `H^{-1}` |
| `3/2` | `|d|`                          | constant | `W_1`-like |
| `2` (default) | `d^2`                  | linear   | `W_2^2`-like |

`eps` is a screening term fixed by `max_pad`: the spatial kernel decays to ~2% at
`2*max_pad` away (screening length `L = max_pad/2`), so gradients are strong out
to ~`L` pixels and directionally correct well beyond, without ghost contamination
wrapping the padded image. `s = 2` (constant → linear gradient) is the robust
default.

```python
from tools.losses import make_sobolev_weight, sobolev_loss

sw = tuple(make_sobolev_weight(H, W, max_pad=1024, s=2.0)
           for H, W in plane_shapes)

def loss_fn(params):
    return sobolev_loss(forward(...), target_signals, sw)   # sums over planes
```

`sobolev_loss` is a **drop-in replacement** for `blur_mse_loss` — same call
signature, same `planes` static argument to select which `(volume, plane)`
indices participate.

## Geomean / log1p plane rebalancing (`sobolev_loss_geomean_log1p`)

`blur_mse_loss` and `sobolev_loss` **sum** the per-plane losses. That is a problem
when planes carry very different charge scales — the collection (Y) plane sees the
full drifted charge while induction (U/V) planes see a smaller bipolar signal, so
after normalization the planes can still differ by an order of magnitude in loss.
A plain sum lets the loudest plane dominate the gradient and the geometry from the
quieter induction planes is drowned out.

The **geometric mean** rebalances automatically: each plane's contribution to the
gradient is weighted by the inverse of its own loss, so a dominant plane is
downweighted and every plane pulls with comparable strength.
`sobolev_loss_geomean_log1p` is the **parameter-free** version, using the
Kolmogorov–Nagumo quasi-arithmetic mean with generator `log1p`:

```python
# tools/losses.py:sobolev_loss_geomean_log1p
log_sum = 0.0
for p in planes:
    lp = sobolev_loss_single(signals_a[p], signals_b[p], spectral_weights[p])
    log_sum = log_sum + jnp.log1p(lp)
return jnp.expm1(log_sum / len(planes))          # prod(1 + L_p)^(1/n) - 1
```

This interpolates between two regimes: for `L_p >> 1` it behaves as a true
geometric mean (scale-normalizing across planes); for `L_p << 1` (near
convergence) it behaves as an arithmetic mean, keeping gradients stable instead of
blowing up as any single `L_p → 0`. The per-plane gradient weight is
`1/(1 + L_p)`, bounded in `[0, 1]` — no `eps` to tune.

```python
from tools.losses import make_sobolev_weight, sobolev_loss_geomean_log1p

sw = tuple(make_sobolev_weight(H, W) for H, W in plane_shapes)

def loss_fn(params):
    return sobolev_loss_geomean_log1p(forward(...), target_signals, sw)
```

!!! note "The eps-tuned variant"
    `sobolev_loss_geomean` is the earlier geometric mean, `exp(mean(log(L_p +
    eps)))`. It needs `eps` set near the converged loss scale to avoid `log(0)`
    and to control the near-convergence behavior. `sobolev_loss_geomean_log1p`
    removes that knob and is preferred.

## Summary

| Function | Weight builder | Plane combine | Params to tune | Status |
|---|---|---|---|---|
| `blur_mse_loss` | `make_spectral_weight` (sum of Gaussians) | sum | `sigmas` ladder | research / original |
| `sobolev_loss` | `make_sobolev_weight` (`1/|f|²ˢ`) | sum | `s`, `max_pad` | research variant |
| `sobolev_loss_geomean` | `make_sobolev_weight` | geometric mean | `s`, `max_pad`, `eps` | research variant |
| **`sobolev_loss_geomean_log1p`** | **`make_sobolev_weight`** | log1p geomean | **`s`, `max_pad`** | **recommended default** |

All four are drop-in compatible: precompute a per-plane weight tuple once, then
call the loss inside your `jax.value_and_grad` closure. See
[optimization](optimization.md) for an end-to-end gradient fit, and
[particle-generation](particle-generation.md) for producing the differentiable
`signals_a` from a muon's start energy and direction.

## References

- Peyré, *Comparison of Distances Between Measures* (2018) — the `H^{-1}` /
  Wasserstein equivalence underlying the Sobolev weight.
