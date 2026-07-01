# Response Kernels

The response stage turns each drift-diffused charge deposit into a small
`(N, kW, kH)` block of induced signal on the readout — the current a moving
electron cloud induces on nearby wires (or pixels) as a function of wire offset
and time. JAXTPC precomputes a **diffusion kernel table** (`DKernel`), then at
runtime interpolates it per deposit. Everything here lives in `tools/kernels.py`.

The runnable companion notebook:
[physics/response_kernels.ipynb](https://github.com/DeepLearnPhysics/JAXTPC/blob/main/notebooks/physics/response_kernels.ipynb).

## The DKernel table

A single base response kernel per plane type (`tools/responses/{U,V,Y}_plane_kernel.npz`)
captures the *field response* — the wire signal from a point charge with **no**
diffusion. Diffusion widens that response, and the width grows with drift
distance. Rather than convolve a Gaussian per deposit at runtime, JAXTPC builds a
table once: the base kernel blurred at `num_s` increasing diffusion levels.

`generate_dkernel_table` produces an array of shape `(num_s, H, W)`:

```python
from tools.kernels import load_response_kernels

kernels = load_response_kernels(
    num_s=32,
    max_sigma_trans_unitless=...,   # max transverse sigma, wire-pitch units
    max_sigma_long_unitless=...,    # max longitudinal sigma, time-bin units
)
DKernel = kernels['Y'].DKernel      # shape (num_s, kernel_height, kernel_width)
```

### How each level is built (reflect-pad + separable Gaussian conv)

For diffusion level `s`, `generate_dkernel_table`:

1. Reflect-pads the base kernel on the leading edge of each axis (`jnp.pad(...,
   mode='reflect')`), so the blur has no wrap-around or truncation artifact at
   the kernel boundary.
2. Applies a **separable 2-D Gaussian convolution** — one 1-D Gaussian along the
   time axis, one along the wire axis — via `jax.lax.conv_general_dilated`.
3. The Gaussian widths scale linearly with the level:
   `sigma_T = sigma_trans_max * s / kernel_dx` (wire axis) and
   `sigma_L = sigma_long_max * s / kernel_dy` (time axis).

The whole ladder of levels is produced with `vmap(make_level)(s_levels)` — no
Python loop over `s`. Gaussian filter size is set from `_N_SIGMAS_BLUR = 5`
(machine-precision truncation).

!!! note "Not a DCT"
    Older docs (and prior `CLAUDE.md` / module-docstring wording) described this
    table as built via a **DCT-domain** operation. **That is incorrect.** The
    table is built by **reflect-padding the base kernel and applying a separable
    2-D Gaussian convolution** (`jax.lax.conv_general_dilated`) at each diffusion
    level — mathematically an exact linear convolution with a truncated Gaussian,
    not a discrete cosine transform. There is no DCT anywhere in `tools/kernels.py`.

## Indexing the table by diffusion level `s`

Diffusion sigma grows as the square root of drift distance, so the table is
indexed by a normalized level

```
s = sqrt(drift_distance_cm / global_max_drift)      # clipped to [0, 1]
```

evaluated per deposit in the response closure (`tools/simulation.py`). `s = 0` is
a deposit at the anode (no diffusion → the raw field response); `s = 1` is a
deposit at the furthest drift (maximum blur). Because the table's Gaussian width
at level `s` is `sigma_max * s`, the `sqrt` mapping makes the realized sigma
scale as `sqrt(drift_distance)` — the correct diffusion law.

!!! warning "s is sqrt-scaled, not linear"
    The level is `s = sqrt(d / d_max)`, **not** `d / d_max`. The table's levels
    are evenly spaced in `s`, so they are *denser* in physical drift distance
    near the anode where the response changes fastest.

### Runtime interpolation

At runtime each deposit's `s` (plus sub-pitch wire and sub-tick time offsets)
selects and interpolates a kernel. `interpolate_diffusion_kernel` does **linear**
interpolation in all three of `(s, wire, time)`:

- `s`: linear blend between the two bracketing table levels
  (`s_idx = floor(s·(num_s-1))`).
- wire: linear blend between adjacent wire bins, offset by the deposit's
  fractional wire position.
- time: linear blend between adjacent time bins (this reduces the output height
  to `kernel_height - 1`).

`apply_diffusion_response` vmaps this over all deposits in a chunk, yielding the
`(N, num_wires, kernel_height-1)` contributions accumulated in `tools/physics.py`.

### num_s and accuracy near the anode

`num_s` defaults to **32** (`create_sim_config`, `load_response_kernels`).
Interpolation error is largest **near the anode** (small `s`), where the response
shape changes most rapidly between levels; raising `num_s` reduces it. The table
is built once, so runtime cost is independent of `num_s` — only the build cost
and memory scale with it. See the memory note on `num_s` for the diffusion-interp
tradeoff (pixel readout benefits from a larger `num_s`).

## Wire vs pixel kernels

The wire and pixel paths use **different kernels with different units**, by
design. This drives the whole downstream units convention.

| | Wire kernel | Pixel kernel |
|---|---|---|
| Loader | `load_response_kernels` | `load_pixel_response_kernel` |
| Table builder | `generate_dkernel_table` (2-D) | `generate_dkernel_table_3d` (3-D) |
| Base kernel shape | `(H, W)` = (time, wire), two-sided | `(Hpy, Hpz, Ht)`, distance-indexed |
| Kernel value | **Dimensionless field-impulse fraction** | **ADC per drift-electron** (chip gain baked in) |
| `intensity × kernel →` | **ENC** (electrons) | **ADC** |
| Downstream chain | response → electronics → noise → digitize → ADC | response → done (no electronics/noise/digitize) |
| Interp routine | `interpolate_diffusion_kernel` | `interpolate_pixel_response_kernel` (+ time rebin) |

The wire NPZ files carry a misleading `units = 'ADC_per_electron'` /
`adc_per_electron` metadata field. **This loader ignores it** and treats the
kernel as a dimensionless field-impulse fraction; the metadata only records the
kernel's first-principles calibration source and is not consumed in the JIT path.
The wire signal only becomes ADC after the electronics → noise → digitize chain.

The pixel kernel additionally **rebins the time axis** from native NPZ resolution
to the simulation time step (`rebin_factor = time_spacing / npz_time_bin`,
averaging groups of fine bins to preserve the current *rate*), and interpolates in
`(s, pixel_y, pixel_z, time)`.

For the full unit accounting and threshold semantics, see [units](units.md). For
the readout-path differences, see [wire vs pixel](../detector/wire-vs-pixel.md).
