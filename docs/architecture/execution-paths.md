# Execution paths

`DetectorSimulator` exposes **three** entry points. All three call into the same
shared physics body in `tools/physics.py` — they differ in what surrounds that
body (batching, post-processing, differentiability) and in a single structural
branch inside `compute_plane_signal`.

| Entry point | Purpose | Loop | `n_actual` | Post-processing | Differentiable |
|---|---|---|---|---|---|
| `process_event` | Production | `fori_loop` | traced (dynamic) | electronics, noise, digitize, track-hits | no |
| `forward` / `forward_segments` | Gradient-based reco/opt | `fori_loop` under `remat` | **static int** | none (wire response only) | **yes** |
| `process_event_light` | Charge + photons only | none (no response) | n/a | none | no |

---

## `process_event` — production

The full detector-response path. Enabled at construction and used for batch
simulation.

```python
response_signals, track_hits, deposits = simulator.process_event(deposits, key=key)
```

- **Batched accumulation.** `compute_plane_signal` runs a `jax.lax.fori_loop`
  over chunks of `response_chunk_size` deposits, giving bounded peak memory.
- **Traced `n_actual`.** The number of real deposits is a **traced** JAX value,
  so the loop bound is computed with `jnp.minimum`. Padding is masked once in
  `compute_volume_physics`, so trailing chunks add zeros harmlessly.
- **Full post-processing** (per the enabled flags): electronics (`electronics_fn`),
  intrinsic noise (`noise_fn`), digitization (`digitize_fn`), and track-hits
  (`track_hits_fn`), all inside the single JIT. Coherent noise, if enabled, is
  applied host-side in NumPy after the JIT returns.
- **Both wire and pixel** readout are supported. Pixel is single-pass: one
  response computation yields both the signal and per-group truth.

Overflow checks (`max_active_buckets`, `max_keys`, `maxg`) raise `RuntimeError`
on the **host** after the JIT returns — see
[capacities](../concepts/capacities.md).

---

## `forward` / `forward_segments` — differentiable

The gradient path, enabled with `differentiable=True, n_segments=N`. Use it to
back-propagate through the physics for reconstruction and optimization.

```python
sim = DetectorSimulator(cfg, differentiable=True, n_segments=20_000)
signals = sim.forward(params, deposits)          # padded local-frame deposits
signals = sim.forward_segments(params, xyz, de, dx)  # global coords, traceable
```

- **`remat` for memory.** The per-volume forward is wrapped in `jax.remat` so
  reverse-mode gradients recompute activations instead of storing them —
  essential for the deep response graph.
- **Static `n_actual`.** In this path `n_actual` is a Python `int` (equal to
  `n_segments` / `total_pad`), so the accumulation loop bound is a compile-time
  constant computed with `min(...)`. A static bound is what makes reverse-mode
  differentiation through the `fori_loop` tractable.
- **Wire-only, response-only.** Construction forces `include_intrinsic_noise`,
  `include_electronics`, `include_track_hits`, `include_digitize`, and
  `use_bucketed` all **off**. There is **no** noise, electronics, digitization,
  or track-hits in this path — just the field response.
- **`forward` vs `forward_segments`.** `forward` expects `DepositData` already in
  local coordinates (it pads to `total_pad`). `forward_segments` takes raw
  **global** positions plus `de`/`dx`, does the local-frame transform and volume
  masking inside the traced function (via `stop_gradient` masks, no NumPy
  splitting), so it can be called directly inside `jax.grad`.

### What carries gradients

Gradients flow through the `SimParams` physics fields:

- `velocity_cm_us`
- `lifetime_us`
- `diffusion_trans_cm2_us`, `diffusion_long_cm2_us`
- `recomb_params` (recombination model parameters)

The differentiable response factory (`_build_response_fn_diff`) regenerates the
`DKernel` diffusion table from the current `SimParams` diffusion values on each
call — with **static** Gaussian filter sizes (`ks_w`/`ks_t`, precomputed from the
default params) so only the widths change, not the graph shape. That is what
lets diffusion be a differentiable parameter.

!!! warning "Pixel is unsupported in the differentiable path"
    Constructing with `differentiable=True` on a pixel-readout detector raises
    `ValueError`. The differentiable forward exists only for wire readout
    (`self.n_segments is not None and self._readout_type == 'wire'`). The pixel
    diff-response factory is a placeholder aliased to the production one.

---

## `process_event_light` — charge + photons only

A stripped path that runs `compute_volume_physics` and returns just the
per-segment charge and scintillation photons — **no wire/pixel response at all**.

```python
deposits = simulator.process_event_light(deposits)
# deposits.volumes[v].charge / .photons now filled
```

Use it when you only need the recombination output (Q and L) and drift-derived
quantities, not the readout signal. It is far cheaper than a full
`process_event` because it skips the entire response/accumulation loop and all
post-processing.

---

## The exact divergence: `compute_plane_signal`

Production and differentiable share `compute_plane_signal` verbatim. The **only**
structural difference at the accumulation level is the loop-bound computation,
gated on the *Python type* of `n_actual`:

```python
if isinstance(n_actual, int):
    # differentiable path — static bound, supports reverse-mode grad
    n_batches = min((n_actual + chunk_size - 1) // chunk_size, max_safe_batches)
else:
    # production path — traced bound
    n_batches = jnp.minimum(
        (n_actual + chunk_size - 1) // chunk_size, max_safe_batches)
```

- **Production** passes `vol_deps.n_actual` — a traced JAX int → `jnp.minimum`.
- **Differentiable** passes `n_segments` — a Python `int` → `min`.

Everything downstream (`compute_chunk_response`, the response `fori_loop`, the
scatter-add) is identical. `compute_plane_signal_bucketed` (and the pixel
bucketed variant) use the same `isinstance` check.

!!! note "One physics body, two callers"
    `compute_volume_physics` → `compute_plane_physics` → `compute_chunk_response`
    → `compute_plane_signal` are called with **identical signatures** from both
    `process_one_volume` (production) and `diff_one_volume` (differentiable).
    There is no separate "differentiable physics" implementation — only the
    wrapper differs (`remat`, static bound, no post-processing). See the
    [reading guide](reading-guide.md).

---

## See also

- [Reading guide](reading-guide.md) — the shared body in execution order.
- [Simulator](simulator.md) — how the closures both paths call are built.
- [Differentiable overview](../differentiable/overview.md) — the gradient path
  in depth: losses, particle generation, optimization.
- [Padding & masking](../concepts/padding-masking.md) — why trailing chunks are
  harmless.
