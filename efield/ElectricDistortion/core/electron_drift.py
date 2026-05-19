"""Trace ionisation electrons through the distorted E-field to the anode."""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import RegularGridInterpolator

from .drift_velocity import drift_velocity


def _build_interpolators(x_grid, y_grid, z_grid, Ex, Ey, Ez):
    """Create RegularGridInterpolators for each field component."""
    kw = dict(bounds_error=False, fill_value=None)
    points = (x_grid, y_grid, z_grid)
    return (
        RegularGridInterpolator(points, Ex, **kw),
        RegularGridInterpolator(points, Ey, **kw),
        RegularGridInterpolator(points, Ez, **kw),
    )


def trace_electron(x0, y0, z0, Ex_interp, Ey_interp, Ez_interp,
                   temperature=89.0, method="RK45",
                   rtol=1e-6, atol=1e-6, t_max=20000.0):
    """Trace one electron from (x0, y0, z0) to the anode (x = 0).

    The electron drifts opposite to the physical E-field (toward x = 0).
    Integration stops when x <= 0.

    Parameters
    ----------
    x0, y0, z0 : float
        True position (cm).
    Ex_interp, Ey_interp, Ez_interp : RegularGridInterpolator
        Interpolators for the three field components (V/cm).
    temperature : float
        LAr temperature (K) for the drift-velocity parameterisation.
    method : str
        ODE solver method (default ``'RK45'``).
    rtol, atol : float
        Tolerances for ``solve_ivp``.
    t_max : float
        Safety cutoff time (us).

    Returns
    -------
    t_drift : float
        Drift time to the anode (us).
    y_anode : float
        Transverse y position where the electron reaches the anode (cm).
    z_anode : float
        Transverse z position where the electron reaches the anode (cm).
    """
    if x0 <= 0.0:
        return 0.0, y0, z0

    def ode_rhs(t, pos):
        pt = np.array([[pos[0], pos[1], pos[2]]])
        ex = float(Ex_interp(pt))
        ey = float(Ey_interp(pt))
        ez = float(Ez_interp(pt))

        e_mag = np.sqrt(ex * ex + ey * ey + ez * ez)
        if e_mag < 1e-10:
            return [0.0, 0.0, 0.0]

        vd = drift_velocity(e_mag, T=temperature)  # cm/us

        # Electron drifts OPPOSITE to physical E-field
        inv_e = vd / e_mag
        return [-inv_e * ex, -inv_e * ey, -inv_e * ez]

    def anode_event(t, pos):
        return pos[0]

    anode_event.terminal = True
    anode_event.direction = -1  # trigger when x decreases through 0

    sol = solve_ivp(
        ode_rhs,
        [0.0, t_max],
        [x0, y0, z0],
        method=method,
        events=anode_event,
        rtol=rtol,
        atol=atol,
        max_step=50.0,
    )

    if sol.t_events[0].size > 0:
        t_drift = float(sol.t_events[0][0])
        y_anode = float(sol.y_events[0][0][1])
        z_anode = float(sol.y_events[0][0][2])
    else:
        # Electron did not reach the anode within t_max — use final state
        t_drift = float(sol.t[-1])
        y_anode = float(sol.y[1, -1])
        z_anode = float(sol.y[2, -1])

    return t_drift, y_anode, z_anode


# ---- parallel helpers (multiprocessing) ------------------------------------

_ex_interp = None
_ey_interp = None
_ez_interp = None
_trace_kw = {}


def _init_worker(x_grid, y_grid, z_grid, Ex, Ey, Ez, kw):
    """Initialise per-worker interpolators (called once per process)."""
    global _ex_interp, _ey_interp, _ez_interp, _trace_kw
    _ex_interp, _ey_interp, _ez_interp = _build_interpolators(
        x_grid, y_grid, z_grid, Ex, Ey, Ez
    )
    _trace_kw = kw


def _trace_one(args):
    """Worker target: trace a single electron."""
    x0, y0, z0 = args
    return trace_electron(x0, y0, z0,
                          _ex_interp, _ey_interp, _ez_interp,
                          **_trace_kw)


def trace_electrons_parallel(output_x, output_y, output_z,
                             x_grid, y_grid, z_grid,
                             Ex, Ey, Ez,
                             temperature=89.0, method="RK45",
                             rtol=1e-6, atol=1e-6, t_max=20000.0,
                             n_workers=None):
    """Trace electrons from every point on the output grid.

    Parameters
    ----------
    output_x, output_y, output_z : 1-D arrays
        Coordinates of the (coarser) output grid.
    x_grid, y_grid, z_grid : 1-D arrays
        Coordinates of the (finer) Poisson-solve grid on which Ex/Ey/Ez
        are defined.
    Ex, Ey, Ez : 3-D arrays
        Electric field components on the Poisson grid (V/cm).
    n_workers : int or None
        Number of parallel processes.  ``None`` → ``os.cpu_count()``.

    Returns
    -------
    t_drift, y_anode, z_anode : 3-D arrays, shape (Nxo, Nyo, Nzo)
    """
    from multiprocessing import Pool, cpu_count

    Nxo = len(output_x)
    Nyo = len(output_y)
    Nzo = len(output_z)

    # Build flat list of (x0, y0, z0) launch points
    mesh = np.meshgrid(output_x, output_y, output_z, indexing="ij")
    points = list(zip(mesh[0].ravel(), mesh[1].ravel(), mesh[2].ravel()))

    kw = dict(temperature=temperature, method=method,
              rtol=rtol, atol=atol, t_max=t_max)

    if n_workers is None:
        n_workers = cpu_count()

    if n_workers <= 1:
        # Serial fallback
        interps = _build_interpolators(x_grid, y_grid, z_grid, Ex, Ey, Ez)
        results = [
            trace_electron(p[0], p[1], p[2], *interps, **kw) for p in points
        ]
    else:
        with Pool(
            processes=n_workers,
            initializer=_init_worker,
            initargs=(x_grid, y_grid, z_grid, Ex, Ey, Ez, kw),
        ) as pool:
            results = pool.map(_trace_one, points)

    # Reshape into 3-D arrays
    results_arr = np.array(results)  # (N, 3)
    shape = (Nxo, Nyo, Nzo)
    t_drift = results_arr[:, 0].reshape(shape)
    y_anode = results_arr[:, 1].reshape(shape)
    z_anode = results_arr[:, 2].reshape(shape)

    return t_drift, y_anode, z_anode
