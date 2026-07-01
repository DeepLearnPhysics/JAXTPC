# Particle Generation

To *fit* a muon — recover its start energy, position, and direction from a
readout — you need to differentiate the whole chain from those track parameters
all the way to the simulated signals. `tools/particle_generator.py` provides the
front end: a **fully differentiable muon generator** that turns a handful of
scalar track parameters into the `(positions_mm, de)` arrays the differentiable
[forward pass](overview.md) consumes. Composed with the simulator and a
[loss](losses.md), it closes the loop `track params → signals → loss` so
`jax.grad` can flow back to the parameters.

The module has two paths, by design:

| Path | Function | Output | Use |
|---|---|---|---|
| **Numpy toy** | `generate_muon_track` / `generate_multiple_tracks` | variable-length, sequential | making synthetic **test data** (host-side, not differentiable) |
| **JAX differentiable** | `generate_muon_segments` / `_trig` | fixed-length, parallel | **optimization** through `jax.grad` |

The numpy path steps a muon forward one `step_size_mm` at a time, looking up
`dE/dx` at the current energy and subtracting it — a simple `while energy >
min_energy` loop that produces a ragged number of steps. It is fine for
generating events to feed the pipeline, but a Python loop of data-dependent length
is not traceable. The differentiable path below replaces it.

## From PDG table to CSDA range

The physics input is the PDG muon stopping-power table for liquid argon
(`tools/data/muon_dedx_lar.csv`). `load_dedx_table_jax` loads it as JAX arrays —
`log(T)` (kinetic energy) and `dE/dx` already converted to MeV/cm by multiplying
the tabulated MeV·cm²/g by the LAr density (1.396 g/cm³).

```python
from tools.particle_generator import load_dedx_table_jax, build_csda_range_table

log_T_table, dedx_table = load_dedx_table_jax()
R_cm, T_MeV = build_csda_range_table(log_T_table, dedx_table)   # (n_points,) each
```

`build_csda_range_table` integrates the reciprocal stopping power on a dense
log-spaced energy grid to get the **CSDA range**

```
R(E) = integral_0^E  1 / (dE/dx)  dE'
```

the total path length a muon of energy `E` travels before stopping. It is
computed once as a trapezoidal cumulative sum and returned as a paired
`(R_cm, T_MeV)` lookup — a monotone map between range and energy that can be
inverted by interpolation in either direction.

## CSDA-range inversion (the differentiable core)

The key idea: instead of sequentially stepping and subtracting energy (which is
`O(N)` and not cleanly differentiable), place all `n_segments` in **parallel**
using the range table. A muon that starts with range `R(E_0)` has, after path
length `s`, remaining range `R(E_0) - s`, so its energy there is
`E(s) = R^{-1}(R(E_0) - s)`. The energy deposited in segment `i` is just the
difference of the inverse-range lookups at its two ends:

```python
# tools/particle_generator._csda_energy_deposits
R_initial = jnp.interp(jnp.log(kinetic_energy_mev), log_T_csda, R_cm_table)
R_at_start = R_initial - indices * step_size_cm          # per-segment, vectorized
R_at_end   = R_initial - (indices + 1) * step_size_cm

# Softplus relaxation at the stopping boundary R_floor -> smooth gradients
R_start_soft = R_floor + jax.nn.softplus((R_at_start - R_floor) / relax) * relax
R_end_soft   = R_floor + jax.nn.softplus((R_at_end   - R_floor) / relax) * relax

E_start = jnp.interp(R_start_soft, R_cm_table, T_MeV_table)   # R^{-1}
E_end   = jnp.interp(R_end_soft,   R_cm_table, T_MeV_table)
return jnp.maximum(E_start - E_end, 0.0)                      # de per segment
```

This is `O(1)` per segment and every operation is a smooth, differentiable
interpolation. The **softplus relaxation** matters at the stopping point: a muon
that ranges out partway along the track hits the table floor `R_floor`, and a hard
clamp there would kill the gradient (and the Bragg-peak energy) for the start
energy `E_0`. Softplus rounds that corner so `d(de)/d(E_0)` stays finite and
correct through the endpoint. `relax_steps` (default 2.0) sets the relaxation
width in units of the step size.

## Generating segments

`generate_muon_segments` assembles positions along a straight line and fills in
the CSDA energies. Direction is a unit vector from spherical angles:

```python
from tools.particle_generator import generate_muon_segments

positions_mm, de = generate_muon_segments(
    kinetic_energy_mev=1000.0,
    start_position_mm=jnp.array([0., 0., 0.]),
    theta=1.2, phi=0.5,
    step_size_mm=3.0,
    n_segments=256,               # STATIC — fixes output length for JIT
    log_T_table=log_T_table,
    dedx_table=dedx_table,
)
# positions_mm: (n_segments, 3) segment centres ; de: (n_segments,) MeV
```

`n_segments` is static (fixed output length → one JIT compilation); segments past
the muon's range simply get `de = 0` from the softplus/`maximum`, so an
over-long track is self-truncating.

### Trig parameterization (`generate_muon_segments_trig`)

For optimization, direction is better parameterized by its **trig components**
`(sin_theta, cos_theta, sin_phi, cos_phi)` rather than the raw angles
`(theta, phi)`. Optimizing angles directly suffers from wrap-around (the loss is
periodic in `phi`, so gradient descent can stall or jump at `±pi`) and from the
polar coordinate singularity. Carrying the sin/cos components as free parameters
and re-normalizing the direction vector avoids both, giving well-conditioned
gradients throughout the fit:

```python
dir_unnorm = jnp.array([sin_theta * cos_phi, sin_theta * sin_phi, cos_theta])
dir_vec = dir_unnorm / jnp.linalg.norm(dir_unnorm)      # robust to un-normalized inputs
```

Everything else (positions, CSDA `de`) is identical to `generate_muon_segments`.
This is the recommended entry point for gradient-based direction fits.

## Composing generation with simulation

`build_muon_forward` wraps the differentiable simulator into a closure over
`(positions_mm, de)`:

```python
from tools.particle_generator import build_muon_forward

# simulator built with differentiable=True, n_segments=n_segments
forward = build_muon_forward(simulator, n_segments=256, step_size_mm=3.0)
signals = forward(positions_mm, de)          # tuple of per-plane arrays
```

`forward` calls `simulator.forward_segments(sim_params, positions_mm, de,
dx=step_size_mm)` — the lightweight segment path that takes **global** positions
and masks/splits them into volumes internally (no host-side numpy splitting, fully
traceable; see [execution paths](../architecture/execution-paths.md)). Chaining
the two gives a single differentiable function from track parameters to signals:

```python
def params_to_signals(ke, start, sin_t, cos_t, sin_p, cos_p):
    pos, de = generate_muon_segments_trig(
        ke, start, sin_t, cos_t, sin_p, cos_p,
        step_size_mm, n_segments, log_T_table, dedx_table)
    de = mask_outside_volume(pos, de, half_extents_mm)
    return forward(pos, de)

def loss_fn(params):
    return sobolev_loss_geomean_log1p(params_to_signals(*params), target, sw)

loss, grads = jax.jit(jax.value_and_grad(loss_fn))(init_params)
```

The signals are in [ENC](../physics/units.md) (the differentiable path is
wire-only). See [losses](losses.md) for the loss and [optimization](optimization.md)
for a full fit loop.

## Volume masking

A generated track can run past the detector edge; those segments must contribute
zero charge. `get_half_extents_mm` reads the per-axis half-extents from the
detector config (converting the YAML ranges from cm to mm), and
`mask_outside_volume` zeros the `de` of any segment whose position falls outside:

```python
from tools.particle_generator import get_half_extents_mm, mask_outside_volume

half_extents_mm = get_half_extents_mm(detector_config)
de = mask_outside_volume(positions_mm, de, half_extents_mm)   # smooth: only de -> 0
```

Masking `de` (rather than dropping segments) keeps the output shape fixed and the
operation differentiable — a segment leaving the volume smoothly loses its
contribution without breaking the trace.

!!! note "Physical scope"
    The differentiable generator models a **single straight muon** with
    continuous-slowing-down energy loss — no multiple scattering, delta rays, or
    secondaries. It is a reconstruction/optimization front end, not a full
    Geant4-style shower simulator. For richer synthetic events, generate with the
    numpy path (or an external simulator) and feed the pipeline directly.
