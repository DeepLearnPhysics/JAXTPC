"""Core physics: charge density, Poisson solver, and E-field computation."""

import numpy as np
from scipy.fft import dstn, idstn


def compute_charge_density(x_grid, y_grid, z_grid, Q, v_ion):
    """Steady-state ion charge density: rho(x) = Q * x / v_ion, uniform in y/z.

    Parameters
    ----------
    x_grid, y_grid, z_grid : 1-D arrays
        Grid coordinates (cm).
    Q : float
        Volumetric charge production rate (C/m^3/s).
    v_ion : float or 1-D array
        Ion drift speed (cm/s).  Scalar for uniform speed, or array of
        shape ``(Nx,)`` for a position-dependent profile.

    Returns
    -------
    rho : ndarray, shape (Nx, Ny, Nz)
        Charge density (C/m^3).
    """
    x_m = x_grid / 100.0        # cm -> m
    v_ion = np.atleast_1d(np.asarray(v_ion, dtype=float))
    v_ion_m = v_ion / 100.0      # cm/s -> m/s
    rho_1d = Q * x_m / v_ion_m   # (C/m^3)

    rho = np.broadcast_to(
        rho_1d[:, None, None],
        (len(x_grid), len(y_grid), len(z_grid)),
    ).copy()
    return rho


def compute_vion_profile(Ex, mu_ion):
    """Compute position-dependent ion drift speed from the x-component of E.

    Parameters
    ----------
    Ex : ndarray, shape (Nx, Ny, Nz)
        x-component of the electric field (V/cm).
    mu_ion : float
        Ion mobility (cm^2/V/s).

    Returns
    -------
    v_ion : ndarray, shape (Nx,)
        Ion drift speed profile (cm/s), averaged over y and z.
    """
    Ex_avg = np.mean(Ex, axis=(1, 2))            # shape (Nx,)
    return mu_ion * np.maximum(Ex_avg, 1.0)       # floor at 1 V/cm


def solve_poisson_dst(rho, Lx, Ly, Lz, epsilon):
    """Solve nabla^2(dphi) = -rho/epsilon with homogeneous Dirichlet BCs.

    Uses a 3-D Discrete Sine Transform (Type I, orthonormal).

    Parameters
    ----------
    rho : ndarray, shape (Nx, Ny, Nz)
        Charge density (C/m^3) on the full grid including boundaries.
    Lx, Ly, Lz : float
        Box dimensions (cm) — converted to metres internally.
    epsilon : float
        Absolute permittivity epsilon_0 * epsilon_r (F/m).

    Returns
    -------
    dphi : ndarray, shape (Nx, Ny, Nz)
        Distortion potential (V), zero on all six faces.
    """
    Lx_m, Ly_m, Lz_m = Lx / 100.0, Ly / 100.0, Lz / 100.0
    Nx, Ny, Nz = rho.shape

    rho_hat = dstn(rho[1:-1, 1:-1, 1:-1], type=1, norm="ortho")

    l = np.arange(1, Nx - 1)
    m = np.arange(1, Ny - 1)
    n = np.arange(1, Nz - 1)

    denom = epsilon * np.pi ** 2 * (
        (l[:, None, None] / Lx_m) ** 2
        + (m[None, :, None] / Ly_m) ** 2
        + (n[None, None, :] / Lz_m) ** 2
    )

    dphi = np.zeros_like(rho)
    dphi[1:-1, 1:-1, 1:-1] = idstn(rho_hat / denom, type=1, norm="ortho")
    return dphi


def compute_efield(dphi, E0, dx, dy, dz):
    """Total *physical* E-field from the distortion potential.

    Physical convention:
        phi_phys(x) = -E0*x + dphi   (cathode at negative HV)
        E = -grad(phi_phys)
        Ex =  E0 - d(dphi)/dx        (positive, points anode -> cathode)
        Ey = -d(dphi)/dy
        Ez = -d(dphi)/dz

    Electrons drift opposite to E  (toward x = 0, the anode).

    Parameters
    ----------
    dphi : ndarray, shape (Nx, Ny, Nz)
        Distortion potential (V).
    E0 : float
        Nominal field magnitude (V/cm).
    dx, dy, dz : float
        Grid spacings (cm).

    Returns
    -------
    Ex, Ey, Ez, E_mag : ndarrays
        Field components and magnitude (V/cm).
    """
    ddphi_dx, ddphi_dy, ddphi_dz = np.gradient(dphi, dx, dy, dz)

    Ex = E0 - ddphi_dx
    Ey = -ddphi_dy
    Ez = -ddphi_dz
    E_mag = np.sqrt(Ex ** 2 + Ey ** 2 + Ez ** 2)

    return Ex, Ey, Ez, E_mag
