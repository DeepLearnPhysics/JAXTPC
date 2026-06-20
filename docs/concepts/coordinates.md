# Coordinates & frames

JAXTPC transforms every deposit into a **volume-local frame** before any physics
runs. Understanding this is essential for extending the simulator (new
detectors, new readout references) — and a frame mismatch here is a real bug
class (see the box at the bottom).

## The local-frame transform

The loader (`tools/loader.py`) maps each deposit from global detector
coordinates to its volume's local frame:

- **Drift axis (x):** `x_local = drift_direction * (x_anode_cm - x_global)`, so
  `x_local ≥ 0` and electrons always drift toward `x_local = 0` (the anode),
  regardless of which way the volume physically drifts.
- **Transverse axes (y, z):** **centered**, `y_local = y_global - y_center`,
  `z_local = z_global - z_center`, where `(y_center, z_center)` is the volume
  center `((min+max)/2)` stored as `yz_center_cm`.

## Why: one canonical frame for all volumes

Because the per-volume `drift_direction`, `x_anode`, and `yz_center` are consumed
*once* in the loader, every volume reduces to the **same** local frame — anode at
`x_local = 0`, drift toward `−x`, transverse centered at 0. The JIT physics body
can then use **fixed constants** instead of indexing per-volume geometry inside
the `lax.scan`/`vmap` over volumes. This is what makes a single compiled body
handle any number of volumes.

The inverse transform (local → global) is applied per-volume when writing output
(`production/save.py`), using that volume's own `drift_direction`, `x_anode`, and
`yz_center` — never volume 0's, never a global assumption.

## The invariant for references

Any quantity you compare against a local position must itself be in the **local**
frame. Two examples in the codebase:

- **Wire indexing** (`tools/geometry.py`) builds the wire index reference from
  `corners_centered = [±half_y, ±half_z]` — centered, i.e. the local frame. ✔
- **Pixel grid origin** (`tools/config.py`) is stored as
  `range_min − center = −half_extent` — also local. ✔

!!! danger "Frame-mismatch bug class"
    If a reference is stored in the **global** frame but compared against
    **local** positions, results are correct only when the volume is centered at
    0 (where the two frames coincide). For an off-center volume, deposits map to
    the wrong cell or out-of-bounds → silent signal loss. This exact bug existed
    in the pixel grid origin (global `range_min` vs local-centered positions) and
    was fixed by storing the origin in the local frame. DUNE ND-LAr (off-center
    pixel modules) is the config that would have triggered it.

## Geometry-uniformity assumption

The JIT body captures `cfg.volumes[0]` and applies its geometry (pitch, origin,
wire spacing/angles, max drift, etc.) to **all** volumes. This is correct only if
all volumes share the same local-frame geometry. Every shipped config satisfies
this (uniform within each detector), but it is currently **unvalidated** — adding
an init-time uniformity check is a recommended guard (see the production-
readiness notes). When adding a detector, keep all volumes geometrically uniform,
or the volume-0 assumption will silently apply.
