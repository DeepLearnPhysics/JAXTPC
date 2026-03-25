"""
Electric field distortion physics for LArTPC simulation.

Provides tools for generating, loading, and applying electric field
distortions and drift corrections from space charge effects (SCE).

Main components
---------------
- ``generate_toy_efield_map`` : Analytic toy E-field with space-charge-like
  distortions (linear longitudinal + transverse edge effects).
  Returns **separate maps per TPC side** to avoid cathode discontinuity.
- ``compute_drift_corrections`` : Numerical Euler integration of electron
  drift paths through a single-side E-field map, producing per-grid-point
  (Δx, Δy, Δz) corrections.
- ``interpolate_map_3d`` : JIT-compatible trilinear interpolation for
  querying a 3D vector field at arbitrary deposit positions.
- ``create_dual_interpolation_fn`` : Builds a JIT-compatible function
  that selects the correct side via ``jnp.where``.

Data layout
-----------
E-field maps : (Nx, Ny, Nz, 3) float32, channels [Ex, Ey, Ez] in V/cm
Drift corrections : (Nx, Ny, Nz, 3) float32, channels [Δx, Δy, Δz] in cm
Grid metadata : origin_cm (3,), spacing_cm (3,)

Each map covers a single TPC side (east: x ∈ [-L, 0], west: x ∈ [0, +L]).
For JIT usage, maps are transposed to (3, Nx, Ny, Nz) channel-first layout
and passed to ``interpolate_map_3d``.
"""

import os
import numpy as np
import jax
import jax.numpy as jnp


# =========================================================================
# JIT-compatible interpolation
# =========================================================================

def interpolate_map_3d(positions_cm, field_map, origin_cm, spacing_cm):
    """
    Trilinear interpolation of a 3D vector field at arbitrary positions.

    Uses ``jax.scipy.ndimage.map_coordinates`` with vmap over the three
    vector components.  Fully JIT-compatible.

    Parameters
    ----------
    positions_cm : jnp.ndarray, shape (N, 3)
        Query positions in cm.
    field_map : jnp.ndarray, shape (3, Nx, Ny, Nz)
        Vector field in **channel-first** layout.
    origin_cm : jnp.ndarray, shape (3,)
        Physical coordinates of the grid origin corner ``[0, 0, 0]``.
    spacing_cm : jnp.ndarray, shape (3,)
        Grid spacing in each dimension (cm).

    Returns
    -------
    jnp.ndarray, shape (N, 3)
        Interpolated vector values at each query position.
    """
    # Physical → fractional grid coordinates
    coords = ((positions_cm - origin_cm) / spacing_cm).T  # (3, N)

    def _interp_one(field_3d):
        return jax.scipy.ndimage.map_coordinates(
            field_3d, coords, order=1, mode='nearest'
        )

    # vmap over the 3 channels (leading axis of field_map)
    return jax.vmap(_interp_one)(field_map).T  # (N, 3)


def create_dual_interpolation_fn(east_map, east_origin, east_spacing,
                                  west_map, west_origin, west_spacing):
    """
    Build a JIT-compatible function that interpolates using per-side maps.

    Queries both side maps and selects the correct result per position
    based on its x coordinate via ``jnp.where``.

    Parameters
    ----------
    east_map, west_map : jnp.ndarray, shape (3, Nx, Ny, Nz)
        Channel-first maps for east (x < 0) and west (x >= 0) sides.
    east_origin, west_origin : jnp.ndarray, shape (3,)
    east_spacing, west_spacing : jnp.ndarray, shape (3,)

    Returns
    -------
    fn : callable
        ``fn(positions_cm) → jnp.ndarray (N, 3)``
    """
    def fn(positions_cm):
        is_east = positions_cm[:, 0] < 0
        east_vals = interpolate_map_3d(
            positions_cm, east_map, east_origin, east_spacing,
        )
        west_vals = interpolate_map_3d(
            positions_cm, west_map, west_origin, west_spacing,
        )
        return jnp.where(is_east[:, None], east_vals, west_vals)

    return fn


# =========================================================================
# Toy E-field generation (per-side)
# =========================================================================

def generate_toy_efield_map(
    half_x_cm,
    half_y_cm,
    half_z_cm,
    nominal_field_Vcm,
    grid_shape=(22, 22, 22),
    epsilon_max=0.05,
    epsilon_trans=0.02,
):
    """
    Generate per-side toy E-field maps with space-charge-like distortions.

    Models a dual-drift TPC with cathode at x = 0 and anodes at
    x = ±half_x.  Each side is returned as a separate map, avoiding the
    sign discontinuity at the cathode.

    Distortion model:

    1. **Longitudinal (Ex)** — linear correction that preserves the
       total voltage integral::

           East  (x ∈ [-L, 0]): Ex = +E0 * [1 + ε * (2x + L) / L]
           West  (x ∈ [0, +L]): Ex = -E0 * [1 + ε * (-2x + L) / L]

       Consistent with Gauss's law for a uniform positive space-charge
       density (dEx/dx > 0 on east side).

    2. **Transverse (Ey, Ez)** — edge-effect correction proportional to
       (y or z) / half_width and distance from anode / L.

    Parameters
    ----------
    half_x_cm, half_y_cm, half_z_cm : float
        Detector half-widths (cm).
    nominal_field_Vcm : float
        Nominal drift field magnitude (V/cm).
    grid_shape : tuple of int
        Grid dimensions (Nx, Ny, Nz) **per side**.
    epsilon_max : float
        Fractional longitudinal distortion.
    epsilon_trans : float
        Fractional transverse distortion.

    Returns
    -------
    east_side : tuple (efield_map, origin_cm, spacing_cm)
        East TPC (x ∈ [-L, 0]).
    west_side : tuple (efield_map, origin_cm, spacing_cm)
        West TPC (x ∈ [0, +L]).
    """
    Nx, Ny, Nz = grid_shape
    E0 = nominal_field_Vcm
    L = half_x_cm

    y = np.linspace(-half_y_cm, half_y_cm, Ny)
    z = np.linspace(-half_z_cm, half_z_cm, Nz)

    spacing_yz = np.array([
        2.0 * half_y_cm / max(Ny - 1, 1),
        2.0 * half_z_cm / max(Nz - 1, 1),
    ], dtype=np.float64)

    # ---- East side: x ∈ [-L, 0] ----
    x_east = np.linspace(-L, 0, Nx)
    XE, YE, ZE = np.meshgrid(x_east, y, z, indexing='ij')

    east_efield = np.zeros((Nx, Ny, Nz, 3), dtype=np.float32)
    east_efield[..., 0] = E0 * (1.0 + epsilon_max * (2.0 * XE + L) / L)

    dist_anode_east = XE + L  # 0 at anode (-L), L at cathode (0)
    frac_east = np.clip(dist_anode_east / L, 0.0, 1.0)
    east_efield[..., 1] = E0 * epsilon_trans * (YE / half_y_cm) * frac_east
    east_efield[..., 2] = E0 * epsilon_trans * (ZE / half_z_cm) * frac_east

    east_origin = np.array([-L, -half_y_cm, -half_z_cm], dtype=np.float64)
    east_spacing = np.array([
        L / max(Nx - 1, 1), spacing_yz[0], spacing_yz[1],
    ], dtype=np.float64)

    # ---- West side: x ∈ [0, +L] ----
    x_west = np.linspace(0, L, Nx)
    XW, YW, ZW = np.meshgrid(x_west, y, z, indexing='ij')

    west_efield = np.zeros((Nx, Ny, Nz, 3), dtype=np.float32)
    west_efield[..., 0] = -E0 * (1.0 + epsilon_max * (-2.0 * XW + L) / L)

    dist_anode_west = L - XW  # 0 at anode (+L), L at cathode (0)
    frac_west = np.clip(dist_anode_west / L, 0.0, 1.0)
    west_efield[..., 1] = E0 * epsilon_trans * (YW / half_y_cm) * frac_west
    west_efield[..., 2] = E0 * epsilon_trans * (ZW / half_z_cm) * frac_west

    west_origin = np.array([0.0, -half_y_cm, -half_z_cm], dtype=np.float64)
    west_spacing = np.array([
        L / max(Nx - 1, 1), spacing_yz[0], spacing_yz[1],
    ], dtype=np.float64)

    return (
        (east_efield, east_origin, east_spacing),
        (west_efield, west_origin, west_spacing),
    )


# =========================================================================
# Path integration: E-field → drift corrections (single side)
# =========================================================================

def compute_drift_corrections(
    efield_map,
    origin_cm,
    spacing_cm,
    anode_x_cm,
    nominal_field_Vcm,
    drift_velocity_cm_us,
    dt_us=1.0,
):
    """
    Compute drift corrections for a single TPC side by integrating
    electron paths through an E-field map.

    For each grid point the electron is propagated to the anode using
    vectorised Euler integration (constant-mobility approximation).

    Parameters
    ----------
    efield_map : np.ndarray, shape (Nx, Ny, Nz, 3)
        E-field [Ex, Ey, Ez] in V/cm for one side.
    origin_cm : np.ndarray, shape (3,)
        Physical coordinates of grid origin.
    spacing_cm : np.ndarray, shape (3,)
        Grid spacing per axis.
    anode_x_cm : float
        x-position of the anode for this side (negative for east,
        positive for west).
    nominal_field_Vcm : float
        Nominal field magnitude (V/cm).
    drift_velocity_cm_us : float
        Nominal drift velocity (cm/μs).
    dt_us : float, optional
        Integration time step (μs).  Default 1.0.

    Returns
    -------
    corrections : np.ndarray, shape (Nx, Ny, Nz, 3)
        Drift corrections [Δx, Δy, Δz] in cm.
    """
    from scipy.interpolate import RegularGridInterpolator

    Nx, Ny, Nz, _ = efield_map.shape
    E0 = nominal_field_Vcm
    v_nom = drift_velocity_cm_us
    mu = v_nom / E0  # constant mobility (cm/μs per V/cm)

    # Grid coordinate arrays
    gx = origin_cm[0] + np.arange(Nx) * spacing_cm[0]
    gy = origin_cm[1] + np.arange(Ny) * spacing_cm[1]
    gz = origin_cm[2] + np.arange(Nz) * spacing_cm[2]

    # Scipy interpolators (one per component)
    interps = [
        RegularGridInterpolator(
            (gx, gy, gz), efield_map[..., i],
            method='linear', bounds_error=False, fill_value=None,
        )
        for i in range(3)
    ]

    # Flatten all grid points
    GX, GY, GZ = np.meshgrid(gx, gy, gz, indexing='ij')
    starts = np.stack([GX.ravel(), GY.ravel(), GZ.ravel()], axis=-1)  # (N, 3)
    N = starts.shape[0]

    nominal_dist = np.abs(starts[:, 0] - anode_x_cm)
    nominal_time = np.where(v_nom > 0, nominal_dist / v_nom, 0.0)

    # Integration state
    pos = starts.copy()
    total_time = np.zeros(N, dtype=np.float64)
    active = nominal_dist > 1e-6  # skip points at the anode

    max_drift = np.max(nominal_dist) if np.any(active) else 0.0
    max_steps = int(3.0 * max_drift / (v_nom * dt_us)) + 10

    # East side: electrons drift in -x, done when pos_x <= anode_x
    # West side: electrons drift in +x, done when pos_x >= anode_x
    drift_negative = anode_x_cm <= 0

    for _ in range(max_steps):
        if not np.any(active):
            break

        idx = np.where(active)[0]
        pts = pos[idx]

        # Interpolate E-field at active positions
        E = np.empty((len(idx), 3))
        for c in range(3):
            E[:, c] = interps[c](pts)

        # Electron velocity: v = -μ * E (drift opposite to field)
        vel = -mu * E

        # Save pre-step positions for crossing interpolation
        pos_before = pos[idx].copy()

        # Euler step
        pos[idx] += vel * dt_us
        total_time[idx] += dt_us

        # Check anode crossing
        if drift_negative:
            done = pos[idx, 0] <= anode_x_cm
        else:
            done = pos[idx, 0] >= anode_x_cm

        if np.any(done):
            done_abs = idx[done]

            # Interpolate exact crossing fraction to remove overshoot
            dx_to_anode = anode_x_cm - pos_before[done, 0]
            dx_per_step = vel[done, 0] * dt_us
            frac = np.where(np.abs(dx_per_step) > 1e-12,
                            dx_to_anode / dx_per_step, 1.0)
            frac = np.clip(frac, 0.0, 1.0)

            pos[done_abs] = pos_before[done] + vel[done] * (frac[:, None] * dt_us)
            total_time[done_abs] -= (1.0 - frac) * dt_us

            active[done_abs] = False

    # Build corrections
    corrections = np.zeros((N, 3), dtype=np.float32)
    corrections[:, 0] = (v_nom * (total_time - nominal_time)).astype(np.float32)
    corrections[:, 1] = (pos[:, 1] - starts[:, 1]).astype(np.float32)
    corrections[:, 2] = (pos[:, 2] - starts[:, 2]).astype(np.float32)

    return corrections.reshape(Nx, Ny, Nz, 3)


# =========================================================================
# Load SCE maps from HDF5
# =========================================================================

_DEFAULT_SCE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), 'config', 'sce_jaxtpc.h5'
)


def load_sce_interpolation_fns(h5_path=_DEFAULT_SCE_PATH):
    """
    Load per-side SCE maps and build JIT-compatible interpolation functions.

    Parameters
    ----------
    h5_path : str
        Path to HDF5 written by ``convert_sce_maps.generate_and_save``.

    Returns
    -------
    sce_efield_fn : callable
        ``fn(positions_cm) -> (N, 3)`` E-field [Ex, Ey, Ez] in V/cm.
    sce_drift_correction_fn : callable
        ``fn(positions_cm) -> (N, 3)`` corrections [dx, dy, dz] in cm.
    """
    from tools.utils import load_sce_data

    sce = load_sce_data(h5_path)

    def _to_jax(side):
        efield = jnp.moveaxis(jnp.array(side['efield_map']), -1, 0)
        corr = jnp.moveaxis(jnp.array(side['drift_correction_map']), -1, 0)
        origin = jnp.array(side['origin_cm'], dtype=jnp.float32)
        spacing = jnp.array(side['spacing_cm'], dtype=jnp.float32)
        return efield, corr, origin, spacing

    e_efield, e_corr, e_origin, e_spacing = _to_jax(sce['east'])
    w_efield, w_corr, w_origin, w_spacing = _to_jax(sce['west'])

    sce_efield_fn = create_dual_interpolation_fn(
        e_efield, e_origin, e_spacing,
        w_efield, w_origin, w_spacing,
    )
    sce_drift_correction_fn = create_dual_interpolation_fn(
        e_corr, e_origin, e_spacing,
        w_corr, w_origin, w_spacing,
    )

    return sce_efield_fn, sce_drift_correction_fn
