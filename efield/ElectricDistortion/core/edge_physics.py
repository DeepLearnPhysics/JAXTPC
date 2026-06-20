"""Valid edge/wall E-field distortions via a general Dirichlet Poisson solve.

The DST solver (``solve_poisson_dst``) assumes δφ = 0 on every face (perfect
field cage). Real near-wall distortions come from *defects* in that boundary
condition — field-cage non-uniformity, resistive-divider mismatch — or from
near-wall charge. This module solves

    ∇²δφ = −ρ/ε ,   δφ = φ_bc  on the boundary

with arbitrary Dirichlet boundary values via a 7-point finite-difference
Laplacian. The result is curl-free by construction (E = E₀x̂ − ∇δφ), so any
edge structure it produces is a *physically valid* electrostatic field — unlike
adding high-frequency wiggles to a fitted distortion map.
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def _lap1d(n, h):
    """1-D second-difference on n interior points (Dirichlet), spacing h."""
    main = -2.0 * np.ones(n)
    off = np.ones(n - 1)
    return sp.diags([off, main, off], [-1, 0, 1]) / h ** 2


def solve_poisson_dirichlet(rho, Lx, Ly, Lz, epsilon, phi_bc):
    """Solve ∇²δφ = −ρ/ε with Dirichlet BC δφ = φ_bc on the box faces.

    Parameters
    ----------
    rho : (Nx,Ny,Nz)   charge density (C/m³).
    Lx,Ly,Lz : float   box dimensions (cm; converted to m internally).
    epsilon : float    absolute permittivity (F/m).
    phi_bc : (Nx,Ny,Nz)   array whose *boundary faces* give the Dirichlet values
                          (V); interior entries are ignored.

    Returns
    -------
    dphi : (Nx,Ny,Nz)   distortion potential (V), equal to φ_bc on the faces.
    """
    Nx, Ny, Nz = rho.shape
    dx = (Lx / 100.0) / (Nx - 1)
    dy = (Ly / 100.0) / (Ny - 1)
    dz = (Lz / 100.0) / (Nz - 1)
    nx, ny, nz = Nx - 2, Ny - 2, Nz - 2

    # interior 3-D Laplacian via Kronecker sums
    Ix, Iy, Iz = (sp.identity(nx), sp.identity(ny), sp.identity(nz))
    L = (sp.kron(sp.kron(_lap1d(nx, dx), Iy), Iz)
         + sp.kron(sp.kron(Ix, _lap1d(ny, dy)), Iz)
         + sp.kron(sp.kron(Ix, Iy), _lap1d(nz, dz))).tocsr()

    # boundary contribution: discrete Laplacian of (boundary-only) phi_bc
    phi_full = np.zeros_like(rho, dtype=float)
    phi_full[0, :, :] = phi_bc[0, :, :];   phi_full[-1, :, :] = phi_bc[-1, :, :]
    phi_full[:, 0, :] = phi_bc[:, 0, :];   phi_full[:, -1, :] = phi_bc[:, -1, :]
    phi_full[:, :, 0] = phi_bc[:, :, 0];   phi_full[:, :, -1] = phi_bc[:, :, -1]
    lap_bc = np.zeros_like(rho, dtype=float)
    lap_bc[1:-1] += (phi_full[2:] - 2 * phi_full[1:-1] + phi_full[:-2]) / dx ** 2
    lap_bc[:, 1:-1] += (phi_full[:, 2:] - 2 * phi_full[:, 1:-1] + phi_full[:, :-2]) / dy ** 2
    lap_bc[:, :, 1:-1] += (phi_full[:, :, 2:] - 2 * phi_full[:, :, 1:-1] + phi_full[:, :, :-2]) / dz ** 2

    rhs = (-rho / epsilon - lap_bc)[1:-1, 1:-1, 1:-1].ravel()
    u = spla.spsolve(L, rhs)

    dphi = phi_full.copy()
    dphi[1:-1, 1:-1, 1:-1] = u.reshape(nx, ny, nz)
    return dphi
