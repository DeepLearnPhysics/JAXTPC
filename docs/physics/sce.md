# Space Charge Effects (SCE)

In a liquid-argon TPC, slow-moving positive argon ions build up a space-charge
density that distorts the drift field. Electrons no longer drift straight along a
uniform field: their arrival **time** shifts and their transverse position
**displaces**. JAXTPC models this per volume as an SCE map queried at each
deposit's position and applied in the drift step. Code lives in
`tools/efield_distortions.py` (maps + interpolation) and `tools/drift.py`
(application).

## The SCE maps

Real detector maps are loaded per volume with `load_sce_per_volume`, which reads
an HDF5 file (`config/sce_jaxtpc.h5` by default) containing a `volume_0`,
`volume_1`, … group per detector volume. Each volume carries two 3-D grids:

- an **E-field map** `(Nx, Ny, Nz, 3)` — the distorted field `[Ex, Ey, Ez]` in
  V/cm, and
- a **drift-correction map** `(Nx, Ny, Nz, 3)` — the per-grid-point corrections.

!!! danger "Channel 0 of the correction map is a drift TIME (µs), not a distance"
    In the drift-correction map, **channel 0 is a drift-*time* delta in µs**
    (`total_time − nominal_time`), and channels 1–2 are the transverse spatial
    displacements `[Δy, Δz]` in cm. **Time is the primary quantity; the corrected
    drift distance is derived from it** (`distance = time × velocity`, in
    `drift.apply_drift_corrections`). This is a genuine footgun: it is easy to
    assume channel 0 is a Δx distance like channels 1–2. It is not. The SCE
    consumer reads it as `t_drift` (see `SCEOutputs.drift_time_corr_us`).

### Per-side maps and the local frame

Maps are stored **per TPC side** to avoid the field sign discontinuity at the
cathode (E flips direction across `x = 0`). When `load_sce_per_volume` is given
the volume geometry, it converts each map into the volume's **local frame**
(anode at `x_local = 0`, y/z centered):

- for a `drift_direction == +1` volume the x-grid is flipped so the anode maps to
  `x_local = 0`;
- the E-field **x-component** is negated (`efield[..., 0] *= -dd`) because it is a
  true vector under axis reversal;
- the correction map's **channel 0 (Δt) is *not* sign-flipped** — it is a scalar
  time, not a vector component — and channels 1–2 (Δy, Δz) are unchanged because
  the y/z axes are not flipped.

Getting these sign rules wrong is exactly the channel-0-is-a-time trap in reverse.

## Interpolation

Both maps are queried with **trilinear interpolation** (`interpolate_map_3d`,
built on `jax.scipy.ndimage.map_coordinates`, `order=1`, `mode='nearest'`),
vmapped over the three field components. `create_single_interpolation_fn` closes
over one volume's map + grid metadata (`origin_cm`, `spacing_cm`) and returns a
JIT-compatible `fn(positions_cm) → (N, 3)`. `load_sce_per_volume` returns one
`(efield_fn, corr_fn)` pair per volume.

## Application in the drift step

For each deposit, the SCE closure interpolates the correction map at the deposit
position, then `drift.apply_drift_corrections` folds it in:

```python
corrected_time     = max(drift_time_us + delta_t_us, 0.0)   # time is primary
corrected_distance = corrected_time * velocity_cm_us        # distance derived
corrected_yz       = positions_yz_cm + delta_yz_cm          # transverse displacement
```

So SCE enters the pipeline as **(a) an E-field distortion** — the interpolated
field feeds the recombination / drift physics — **and (b) a spatial + temporal
displacement** of where and when the charge arrives. The corrected drift distance
then flows on into the diffusion level `s` (see [response kernels](response-kernels.md))
and the corrected `(y, z)` into wire/pixel projection.

## Differentiable SCE

A differentiable SCE path exists: the same interpolation is JIT/`grad`-friendly
(`interpolate_map_3d` is built from JAX primitives), so an SCE field can be
carried as a parameter and optimized rather than fixed. This is the basis for the
learned per-module SCE field work.

## Toy / utility generators

For testing and map authoring (no real detector map required),
`tools/efield_distortions.py` provides:

- `generate_toy_efield_map` — an analytic space-charge-like field with a linear
  longitudinal distortion plus transverse edge effects, returned as **separate
  east/west side maps** (again, to avoid the cathode discontinuity).
- `compute_drift_corrections` — numerically integrates electron drift paths
  through a single-side E-field map (vectorized Euler integration to the anode),
  producing the correction map in the same layout as the loaded maps: **channel 0
  = Δt (µs)**, channels 1–2 = Δy, Δz (cm).

These are utilities/toy generators, not part of the default forward pass; the
production path loads real maps via `load_sce_per_volume`.
