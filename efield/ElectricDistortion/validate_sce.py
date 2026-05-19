#!/usr/bin/env python3
"""Validate SCE simulation results: boundary conditions, symmetry, magnitudes.

Usage
-----
    python -m ElectricDistortion.validate_sce --detector jaxtpc --quick
"""

import argparse
import sys

import numpy as np
import matplotlib.pyplot as plt

from .io.config_loader import build_params
from .run_sce import run


# ──────────────────────────────────────────────────────────────────────────── #
#  Validation checks                                                          #
# ──────────────────────────────────────────────────────────────────────────── #

def _check(name, condition, detail=""):
    tag = "PASS" if condition else "FAIL"
    msg = f"  [{tag}] {name}"
    if detail:
        msg += f"  ({detail})"
    print(msg)
    return condition


def validate(results):
    """Run all validation checks.  Returns True if everything passes."""
    p = results["params"]
    E0 = p["E0"]
    Lx, Ly, Lz = p["Lx"], p["Ly"], p["Lz"]

    ox = results["output_x"]
    oy = results["output_y"]
    oz = results["output_z"]
    dx = results["delta_x"]
    dy = results["delta_y"]
    dz = results["delta_z"]

    xp = results["x_poisson"]
    yp = results["y_poisson"]
    zp = results["z_poisson"]
    Ex = results["Ex"]
    Ey = results["Ey"]
    Ez = results["Ez"]
    E_mag = results["E_mag"]
    E_ratio = results["E_ratio"]
    dphi = results["dphi"]

    all_pass = True

    # ── 1. Boundary conditions on dphi ────────────────────────────────────
    print("\n── Poisson BC (dphi = 0 on all 6 faces) ──")
    for label, sl in [
        ("x=0",  np.s_[0, :, :]),
        ("x=Lx", np.s_[-1, :, :]),
        ("y=0",  np.s_[:, 0, :]),
        ("y=Ly", np.s_[:, -1, :]),
        ("z=0",  np.s_[:, :, 0]),
        ("z=Lz", np.s_[:, :, -1]),
    ]:
        val = np.max(np.abs(dphi[sl]))
        all_pass &= _check(f"dphi({label}) == 0", val < 1e-10,
                            f"max |dphi| = {val:.2e}")

    # ── 2. Distortion at anode (x=0) ──────────────────────────────────────
    print("\n── Distortion at anode (x = 0): should be zero ──")
    all_pass &= _check("delta_x(x=0) == 0",
                        np.allclose(dx[0, :, :], 0, atol=1e-6),
                        f"max = {np.max(np.abs(dx[0,:,:])):.2e}")
    all_pass &= _check("delta_y(x=0) == 0",
                        np.allclose(dy[0, :, :], 0, atol=1e-6),
                        f"max = {np.max(np.abs(dy[0,:,:])):.2e}")
    all_pass &= _check("delta_z(x=0) == 0",
                        np.allclose(dz[0, :, :], 0, atol=1e-6),
                        f"max = {np.max(np.abs(dz[0,:,:])):.2e}")

    # ── 3. E-field: Gauss's law → Ex increases along x ────────────────────
    print("\n── E-field monotonicity (positive ρ → dEx/dx > 0) ──")
    iy_mid = len(yp) // 2
    iz_mid = len(zp) // 2
    ex_profile = Ex[:, iy_mid, iz_mid]
    mono = np.all(np.diff(ex_profile) >= -1e-6)
    all_pass &= _check("Ex increases along x at centre",
                        mono,
                        f"Ex(x=0)={ex_profile[0]:.2f}, "
                        f"Ex(x=Lx)={ex_profile[-1]:.2f} V/cm")

    # ── 4. E-field ratio range ────────────────────────────────────────────
    print("\n── E-field ratio |E|/E0 ──")
    all_pass &= _check("|E|/E0 at anode < 1 (field weakened by SCE)",
                        E_ratio[0, iy_mid, iz_mid] < 1.0,
                        f"|E|/E0 = {E_ratio[0, iy_mid, iz_mid]:.4f}")
    all_pass &= _check("|E|/E0 at cathode > 1 (field enhanced by SCE)",
                        E_ratio[-1, iy_mid, iz_mid] > 1.0,
                        f"|E|/E0 = {E_ratio[-1, iy_mid, iz_mid]:.4f}")
    all_pass &= _check("|E|/E0 overall range reasonable (0.5-2.0)",
                        E_ratio.min() > 0.5 and E_ratio.max() < 2.0,
                        f"[{E_ratio.min():.4f}, {E_ratio.max():.4f}]")

    # ── 5. Symmetry ───────────────────────────────────────────────────────
    print("\n── Symmetry (charge density uniform in y, z) ──")

    # E_ratio symmetric about y-midplane
    e_y_sym = np.allclose(E_ratio[:, :, :], E_ratio[:, ::-1, :],
                          rtol=1e-4, atol=1e-4)
    all_pass &= _check("|E|/E0 symmetric about y-midplane", e_y_sym)

    # E_ratio symmetric about z-midplane
    e_z_sym = np.allclose(E_ratio[:, :, :], E_ratio[:, :, ::-1],
                          rtol=1e-4, atol=1e-4)
    all_pass &= _check("|E|/E0 symmetric about z-midplane", e_z_sym)

    # delta_y antisymmetric about y-midplane
    nyo = len(oy)
    if nyo % 2 == 1:
        dy_asym = np.allclose(dy[:, :nyo // 2, :],
                              -dy[:, nyo - 1:nyo // 2:-1, :],
                              rtol=1e-2, atol=0.5)
        all_pass &= _check("delta_y antisymmetric about y-midplane",
                            dy_asym,
                            f"max mismatch = "
                            f"{np.max(np.abs(dy[:,:nyo//2,:] + dy[:,nyo-1:nyo//2:-1,:])):.3f} cm")

    # delta_y at y-midplane should be ~0
    iy_out_mid = nyo // 2
    dy_mid = np.max(np.abs(dy[:, iy_out_mid, :]))
    all_pass &= _check("delta_y at y-midplane ≈ 0 (by symmetry)",
                        dy_mid < 1.0,
                        f"max |delta_y| = {dy_mid:.4f} cm")

    # delta_z antisymmetric about z-midplane
    nzo = len(oz)
    if nzo % 2 == 1:
        dz_asym = np.allclose(dz[:, :, :nzo // 2],
                              -dz[:, :, nzo - 1:nzo // 2:-1],
                              rtol=1e-2, atol=0.5)
        all_pass &= _check("delta_z antisymmetric about z-midplane",
                            dz_asym,
                            f"max mismatch = "
                            f"{np.max(np.abs(dz[:,:,:nzo//2] + dz[:,:,nzo-1:nzo//2:-1])):.3f} cm")

    # ── 6. delta_x: longitudinal distortion ─────────────────────────────
    print("\n── delta_x: longitudinal distortion ──")
    all_pass &= _check("delta_x >= 0 at centre (y-z midplane)",
                        np.all(dx[:, nyo // 2, nzo // 2] >= -0.01),
                        f"min = {dx[:, nyo//2, nzo//2].min():.4f} cm")
    dx_centre = dx[:, nyo // 2, nzo // 2]
    # delta_x should peak in the bulk then can decrease near cathode
    # (Dirichlet BC → dphi=0 at cathode → field returns to E0)
    peak_idx = np.argmax(dx_centre)
    all_pass &= _check("delta_x peak is in interior (not at anode)",
                        peak_idx > 0,
                        f"peak at x = {ox[peak_idx]:.1f} cm, "
                        f"value = {dx_centre[peak_idx]:.3f} cm")
    all_pass &= _check("delta_x(anode) ≈ 0 and delta_x(cathode) > 0",
                        abs(dx_centre[0]) < 0.01 and dx_centre[-1] > 0,
                        f"dx(0) = {dx_centre[0]:.4f}, "
                        f"dx(Lx) = {dx_centre[-1]:.3f} cm")

    # ── 7. Transverse distortions: largest at edges, zero at midplane ─────
    print("\n── Transverse distortions ──")
    # delta_y should be largest near y-boundary, far from anode
    dy_max_abs = np.max(np.abs(dy))
    all_pass &= _check("delta_y has non-zero transverse distortion",
                        dy_max_abs > 0.01,
                        f"max |delta_y| = {dy_max_abs:.3f} cm")

    dz_max_abs = np.max(np.abs(dz))
    all_pass &= _check("delta_z has non-zero transverse distortion",
                        dz_max_abs > 0.01,
                        f"max |delta_z| = {dz_max_abs:.3f} cm")

    # Since Ly == Lz for jaxtpc, delta_y and delta_z magnitudes should match
    if abs(Ly - Lz) < 1.0:
        mag_match = abs(dy_max_abs - dz_max_abs) / max(dy_max_abs, dz_max_abs) < 0.05
        all_pass &= _check("Ly ≈ Lz → |delta_y|_max ≈ |delta_z|_max",
                            mag_match,
                            f"|dy|={dy_max_abs:.3f}, |dz|={dz_max_abs:.3f}")

    # ── 8. Ey, Ez at boundaries ───────────────────────────────────────────
    print("\n── Transverse field components at boundaries ──")
    # On y and z boundaries, Ey and Ez can be non-zero (Neumann not enforced),
    # but far from edges in x, the transverse fields should be small
    ey_centre_x = np.max(np.abs(Ey[len(xp) // 2, len(yp) // 2, :]))
    all_pass &= _check("Ey small at (x_mid, y_mid, z)",
                        ey_centre_x < 0.05 * E0,
                        f"max |Ey| = {ey_centre_x:.3f} V/cm "
                        f"({ey_centre_x / E0 * 100:.2f}% of E0)")

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    if all_pass:
        print("ALL CHECKS PASSED")
    else:
        print("SOME CHECKS FAILED — review output above")
    print("=" * 60)
    return all_pass


# ──────────────────────────────────────────────────────────────────────────── #
#  Improved diagnostic plots                                                  #
# ──────────────────────────────────────────────────────────────────────────── #

def _mid(arr):
    return int(np.argmin(np.abs(arr - (arr[0] + arr[-1]) / 2.0)))


def _quarter(arr):
    """Index closest to the 25% mark — away from midplane, away from edge."""
    target = arr[0] + 0.25 * (arr[-1] - arr[0])
    return int(np.argmin(np.abs(arr - target)))


def plot_validation(results, save_path="sce_validation.png"):
    """Produce a 3x3 diagnostic figure with meaningful slices."""
    p = results["params"]
    E0 = p["E0"]

    xp = results["x_poisson"]
    yp = results["y_poisson"]
    zp = results["z_poisson"]
    E_mag = results["E_mag"]
    E_ratio = results["E_ratio"]

    ox = results["output_x"]
    oy = results["output_y"]
    oz = results["output_z"]
    dx = results["delta_x"]
    dy = results["delta_y"]
    dz = results["delta_z"]

    fig, axes = plt.subplots(3, 3, figsize=(20, 16))

    # ── Row 0: E-field ratio ──────────────────────────────────────────────
    # (0,0) |E|/E0 in xz plane at mid-y
    iy = _mid(yp)
    data = E_ratio[:, iy, :].T
    ext = [xp[0], xp[-1], zp[0], zp[-1]]
    im = axes[0, 0].imshow(data, origin="lower", extent=ext, aspect="auto",
                            cmap="RdBu_r", vmin=0.8, vmax=1.4)
    plt.colorbar(im, ax=axes[0, 0], label="|E| / E0")
    axes[0, 0].set_xlabel("x (cm)")
    axes[0, 0].set_ylabel("z (cm)")
    axes[0, 0].set_title(f"|E|/E0  xz-plane (y = {yp[iy]:.0f} cm)")

    # (0,1) |E|/E0 in xy plane at mid-z
    iz = _mid(zp)
    data = E_ratio[:, :, iz].T
    ext = [xp[0], xp[-1], yp[0], yp[-1]]
    im = axes[0, 1].imshow(data, origin="lower", extent=ext, aspect="auto",
                            cmap="RdBu_r", vmin=0.8, vmax=1.4)
    plt.colorbar(im, ax=axes[0, 1], label="|E| / E0")
    axes[0, 1].set_xlabel("x (cm)")
    axes[0, 1].set_ylabel("y (cm)")
    axes[0, 1].set_title(f"|E|/E0  xy-plane (z = {zp[iz]:.0f} cm)")

    # (0,2) |E|/E0 profiles along x at centre and corners
    iy_m, iz_m = _mid(yp), _mid(zp)
    axes[0, 2].plot(xp, E_ratio[:, iy_m, iz_m], "b-", lw=2, label="centre")
    axes[0, 2].plot(xp, E_ratio[:, 0, 0], "r--", lw=1, alpha=0.7,
                    label="corner (0,0)")
    axes[0, 2].plot(xp, E_ratio[:, 0, iz_m], "g--", lw=1, alpha=0.7,
                    label="edge (0, mid)")
    axes[0, 2].axhline(1.0, ls=":", color="gray", lw=0.8)
    axes[0, 2].set_xlabel("x (cm)")
    axes[0, 2].set_ylabel("|E| / E0")
    axes[0, 2].set_title("|E|/E0 along x (multiple lines)")
    axes[0, 2].legend(fontsize=8)

    # ── Row 1: Distortion maps in xz (mid-y) ─────────────────────────────
    iy_o = _mid(oy)
    iz_o = _mid(oz)
    ext_o = [ox[0], ox[-1], oz[0], oz[-1]]

    # (1,0) delta_x in xz at mid-y
    data = dx[:, iy_o, :].T
    vmax = max(abs(np.nanmin(data)), abs(np.nanmax(data)), 0.01)
    im = axes[1, 0].imshow(data, origin="lower", extent=ext_o, aspect="auto",
                            cmap="Reds", vmin=0, vmax=vmax)
    plt.colorbar(im, ax=axes[1, 0], label="cm")
    axes[1, 0].set_xlabel("x (cm)")
    axes[1, 0].set_ylabel("z (cm)")
    axes[1, 0].set_title(f"delta_x  xz-plane (y = {oy[iy_o]:.0f} cm)")

    # (1,1) delta_y in xy at mid-z  ← the key fix: show in xy plane
    data = dy[:, :, iz_o].T
    vmax = max(abs(np.nanmin(data)), abs(np.nanmax(data)), 0.01)
    im = axes[1, 1].imshow(data, origin="lower",
                            extent=[ox[0], ox[-1], oy[0], oy[-1]],
                            aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    plt.colorbar(im, ax=axes[1, 1], label="cm")
    axes[1, 1].set_xlabel("x (cm)")
    axes[1, 1].set_ylabel("y (cm)")
    axes[1, 1].set_title(f"delta_y  xy-plane (z = {oz[iz_o]:.0f} cm)")

    # (1,2) delta_z in xz at mid-y
    data = dz[:, iy_o, :].T
    vmax = max(abs(np.nanmin(data)), abs(np.nanmax(data)), 0.01)
    im = axes[1, 2].imshow(data, origin="lower", extent=ext_o, aspect="auto",
                            cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    plt.colorbar(im, ax=axes[1, 2], label="cm")
    axes[1, 2].set_xlabel("x (cm)")
    axes[1, 2].set_ylabel("z (cm)")
    axes[1, 2].set_title(f"delta_z  xz-plane (y = {oy[iy_o]:.0f} cm)")

    # ── Row 2: Profiles along x ──────────────────────────────────────────
    # (2,0) delta_x vs x at centre and at 25%/75% offsets
    axes[2, 0].plot(ox, dx[:, iy_o, iz_o], "b-", lw=2,
                    label=f"y={oy[iy_o]:.0f}, z={oz[iz_o]:.0f}")
    iy_q = _quarter(oy)
    iz_q = _quarter(oz)
    axes[2, 0].plot(ox, dx[:, iy_q, iz_q], "r--", lw=1,
                    label=f"y={oy[iy_q]:.0f}, z={oz[iz_q]:.0f}")
    axes[2, 0].plot(ox, dx[:, 0, 0], "g:", lw=1,
                    label=f"y={oy[0]:.0f}, z={oz[0]:.0f} (corner)")
    axes[2, 0].axhline(0, ls=":", color="gray", lw=0.5)
    axes[2, 0].set_xlabel("x (cm)")
    axes[2, 0].set_ylabel("delta_x (cm)")
    axes[2, 0].set_title("delta_x vs x")
    axes[2, 0].legend(fontsize=8)

    # (2,1) delta_y vs y at several x positions (at mid-z)
    ix_vals = [len(ox) // 4, len(ox) // 2, 3 * len(ox) // 4, -1]
    for ix in ix_vals:
        axes[2, 1].plot(oy, dy[ix, :, iz_o], label=f"x={ox[ix]:.0f} cm")
    axes[2, 1].axhline(0, ls=":", color="gray", lw=0.5)
    axes[2, 1].set_xlabel("y (cm)")
    axes[2, 1].set_ylabel("delta_y (cm)")
    axes[2, 1].set_title("delta_y vs y (at various x)")
    axes[2, 1].legend(fontsize=8)

    # (2,2) delta_z vs z at several x positions (at mid-y)
    for ix in ix_vals:
        axes[2, 2].plot(oz, dz[ix, iy_o, :], label=f"x={ox[ix]:.0f} cm")
    axes[2, 2].axhline(0, ls=":", color="gray", lw=0.5)
    axes[2, 2].set_xlabel("z (cm)")
    axes[2, 2].set_ylabel("delta_z (cm)")
    axes[2, 2].set_title("delta_z vs z (at various x)")
    axes[2, 2].legend(fontsize=8)

    fig.suptitle(
        f"SCE Validation: {p.get('Lx', 0):.0f}x{p.get('Ly', 0):.0f}"
        f"x{p.get('Lz', 0):.0f} cm,  E0={E0:.0f} V/cm,  "
        f"Q={p['Q_charge_production']:.1e} C/m³/s",
        fontsize=13, y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"\nValidation plot saved to {save_path}")
    return fig


# ──────────────────────────────────────────────────────────────────────────── #
#  CLI                                                                        #
# ──────────────────────────────────────────────────────────────────────────── #

def main():
    parser = argparse.ArgumentParser(description="Validate SCE results")
    parser.add_argument("--detector", type=str, default=None)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--Nx", type=int, default=None)
    parser.add_argument("--Nxo", type=int, default=None)
    parser.add_argument("--output-plot", type=str, default="sce_validation.png")
    args = parser.parse_args()

    overrides = {}
    if args.quick:
        overrides.update(Nx_poisson=51, Ny_poisson=51, Nz_poisson=51,
                         Nx_output=11, Ny_output=11, Nz_output=11)
    if args.Nx:
        overrides.update(Nx_poisson=args.Nx, Ny_poisson=args.Nx,
                         Nz_poisson=args.Nx)
    if args.Nxo:
        overrides.update(Nx_output=args.Nxo, Ny_output=args.Nxo,
                         Nz_output=args.Nxo)

    params = build_params(preset=args.detector, overrides=overrides)
    print("Running SCE simulation ...")
    results = run(params)

    validate(results)
    plot_validation(results, save_path=args.output_plot)


if __name__ == "__main__":
    main()
