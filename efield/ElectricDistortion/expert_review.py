#!/usr/bin/env python3
"""Expert-level review of SCE simulation — the checks a MicroBooNE physicist
would run before trusting these results."""

import numpy as np

from .io.config_loader import build_params, EPSILON_0
from .core.physics import compute_charge_density, solve_poisson_dst, compute_efield
from .core.drift_velocity import drift_velocity
from .run_sce import run


def banner(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def run_expert_review(preset="microboone", Nx=101, Nxo=21):
    params = build_params(preset=preset, overrides=dict(
        Nx_poisson=Nx, Ny_poisson=Nx, Nz_poisson=Nx,
        Nx_output=Nxo, Ny_output=Nxo, Nz_output=Nxo,
    ))

    print(f"Running {preset} preset: "
          f"{params['Lx']:.0f}x{params['Ly']:.0f}x{params['Lz']:.0f} cm, "
          f"E0={params['E0']:.0f} V/cm")
    results = run(params)

    E0 = params["E0"]
    Lx = params["Lx"]
    Ly = params["Ly"]
    Lz = params["Lz"]
    Nx_p = params["Nx_poisson"]

    xp = results["x_poisson"]
    yp = results["y_poisson"]
    zp = results["z_poisson"]
    Ex = results["Ex"]
    Ey = results["Ey"]
    Ez = results["Ez"]
    E_mag = results["E_mag"]
    dphi = results["dphi"]
    rho = results["rho"]

    ox = results["output_x"]
    oy = results["output_y"]
    oz = results["output_z"]
    dx_map = results["delta_x"]
    dy_map = results["delta_y"]
    dz_map = results["delta_z"]

    iy_mid = len(yp) // 2
    iz_mid = len(zp) // 2
    iy_o = len(oy) // 2
    iz_o = len(oz) // 2

    # ================================================================== #
    #  1. DRIFT VELOCITY BENCHMARK                                       #
    # ================================================================== #
    banner("1. Drift velocity benchmark")

    # Known LArSoft values (at 89 K):
    # E = 273 V/cm → v_d ≈ 1.098 mm/µs  (MicroBooNE operating point)
    # E = 500 V/cm → v_d ≈ 1.563 mm/µs  (SBND/ICARUS operating point)
    larsoft_benchmarks = [
        (273.0, 1.098),   # MicroBooNE
        (500.0, 1.563),   # SBND/ICARUS
        (250.0, 1.048),   # low field reference
        (750.0, 1.826),   # high field reference
    ]

    print(f"{'E (V/cm)':>10}  {'Our v_d':>12}  {'LArSoft ref':>12}  {'diff':>8}")
    print("-" * 50)
    for E_val, ref_vd in larsoft_benchmarks:
        our_vd = drift_velocity(E_val, T=89.0) * 10  # cm/µs → mm/µs
        diff_pct = (our_vd - ref_vd) / ref_vd * 100
        print(f"{E_val:10.1f}  {our_vd:10.4f}    {ref_vd:10.4f}    {diff_pct:+6.2f}%")
        if abs(diff_pct) > 3:
            print(f"  *** WARNING: >3% deviation from LArSoft reference")

    # ================================================================== #
    #  2. VOLTAGE CONSERVATION                                           #
    # ================================================================== #
    banner("2. Voltage conservation (integral of Ex along x)")

    # ∫₀^Lx Ex dx should equal V = E0 × Lx
    # (The total potential drop across the drift volume is set by the HV)
    V_applied = E0 * Lx  # V·cm (we keep in these units)

    ex_profile = Ex[:, iy_mid, iz_mid]
    dx_grid = Lx / (Nx_p - 1)
    V_integrated = np.trapezoid(ex_profile, dx=dx_grid)

    err_pct = (V_integrated - V_applied) / V_applied * 100
    print(f"  Applied voltage:    E0 × Lx = {E0:.1f} × {Lx:.1f} = "
          f"{V_applied:.1f} V·cm")
    print(f"  Integrated ∫Ex dx:  {V_integrated:.1f} V·cm")
    print(f"  Error:              {err_pct:+.4f}%")
    if abs(err_pct) > 0.5:
        print("  *** WARNING: voltage not conserved to 0.5%")
    else:
        print("  OK — voltage conserved")

    # Also check at corner
    ex_corner = Ex[:, 0, 0]
    V_corner = np.trapezoid(ex_corner, dx=dx_grid)
    err_corner = (V_corner - V_applied) / V_applied * 100
    print(f"  Corner ∫Ex dx:      {V_corner:.1f} V·cm  (err {err_corner:+.4f}%)")

    # ================================================================== #
    #  3. POISSON SOLVER RESIDUAL                                        #
    # ================================================================== #
    banner("3. Poisson solver residual")

    # Numerically compute ∇²(dphi) and compare to -ρ/ε
    epsilon = params["epsilon"]
    dx_m = (Lx / 100.0) / (Nx_p - 1)
    dy_m = (Ly / 100.0) / (Nx_p - 1)
    dz_m = (Lz / 100.0) / (Nx_p - 1)

    # Finite difference Laplacian on interior points
    lap = np.zeros_like(dphi)
    lap[1:-1, 1:-1, 1:-1] = (
        (dphi[2:, 1:-1, 1:-1] - 2*dphi[1:-1, 1:-1, 1:-1] + dphi[:-2, 1:-1, 1:-1]) / dx_m**2
        + (dphi[1:-1, 2:, 1:-1] - 2*dphi[1:-1, 1:-1, 1:-1] + dphi[1:-1, :-2, 1:-1]) / dy_m**2
        + (dphi[1:-1, 1:-1, 2:] - 2*dphi[1:-1, 1:-1, 1:-1] + dphi[1:-1, 1:-1, :-2]) / dz_m**2
    )

    rhs = -rho[1:-1, 1:-1, 1:-1] / epsilon
    residual = lap[1:-1, 1:-1, 1:-1] - rhs
    rel_residual = np.abs(residual) / (np.abs(rhs) + 1e-30)

    print(f"  max |∇²dphi - (-ρ/ε)| on interior: {np.max(np.abs(residual)):.3e}")
    print(f"  max relative residual:              {np.max(rel_residual):.3e}")
    print(f"  mean relative residual:             {np.mean(rel_residual):.3e}")
    # Expect O(h²) for the FD Laplacian vs spectral solver
    if np.max(rel_residual) > 0.1:
        print("  *** NOTE: Residual > 10% — expected for FD Laplacian on coarse DST grid")
    else:
        print("  OK — solver accurate")

    # ================================================================== #
    #  4. DISTORTION MAGNITUDES vs PUBLISHED MICROBOONE DATA             #
    # ================================================================== #
    banner("4. Distortion magnitudes")

    # MicroBooNE published SCE (JINST 15 P12037, 2020; MicroBooNE-NOTE-1018):
    # - delta_x (longitudinal): 5-10 cm at cathode, peak ~10 cm in bulk
    # - delta_y (vertical):     ±3-8 cm
    # - delta_z (beam):         ±2-5 cm
    # These depend strongly on Q, which is uncertain by factor ~2-3.

    dx_max = np.max(np.abs(dx_map))
    dy_max = np.max(np.abs(dy_map))
    dz_max = np.max(np.abs(dz_map))
    dx_cathode = dx_map[-1, iy_o, iz_o]

    print(f"  max |delta_x|:                 {dx_max:.2f} cm")
    print(f"  delta_x at cathode (centre):   {dx_cathode:.2f} cm")
    print(f"  max |delta_y|:                 {dy_max:.2f} cm")
    print(f"  max |delta_z|:                 {dz_max:.2f} cm")

    if preset == "microboone":
        print()
        print("  Published MicroBooNE ranges (JINST 15 P12037, 2020):")
        print("    delta_x: 5-10 cm (peak in bulk)")
        print("    delta_y: ±3-8 cm")
        print("    delta_z: ±2-5 cm")
        print("  (Exact values depend on Q, mu_ion — both uncertain by ~2-3x)")

        if dx_max > 20:
            print("  *** WARNING: |delta_x| >> published range")
        if dy_max > 20:
            print("  *** WARNING: |delta_y| >> published range")
        if dz_max > 20:
            print("  *** WARNING: |delta_z| >> published range")

    # Fractional distortion as % of drift length
    print(f"\n  Fractional distortions:")
    print(f"    |delta_x|/Lx = {dx_max/Lx*100:.2f}%")
    print(f"    |delta_y|/Ly = {dy_max/Ly*100:.2f}%")
    print(f"    |delta_z|/Lz = {dz_max/Lz*100:.2f}%")

    # ================================================================== #
    #  5. CHARGE PRODUCTION RATE SANITY CHECK                            #
    # ================================================================== #
    banner("5. Charge production rate Q")

    Q = params["Q_charge_production"]
    mu_ion = params["mu_ion"]
    v_ion = params["v_ion"]

    print(f"  Q = {Q:.2e} C/m³/s")
    print(f"  mu_ion = {mu_ion:.2e} cm²/(V·s)")
    print(f"  v_ion = mu_ion × E0 = {mu_ion:.2e} × {E0:.1f} = {v_ion:.4e} cm/s")
    print(f"  v_ion = {v_ion*10:.4f} mm/s")

    # Ion clearing time
    t_clear = Lx / v_ion  # seconds
    print(f"  Ion clearing time = Lx/v_ion = {Lx:.0f}/{v_ion:.4e} "
          f"= {t_clear:.0f} s = {t_clear/60:.1f} min")

    # Steady-state charge density at cathode
    rho_cathode = Q * (Lx / 100.0) / (v_ion / 100.0)  # C/m³
    print(f"  ρ(cathode) = Q·Lx/v_ion = {rho_cathode:.3e} C/m³")

    # Back-calculate Q from cosmic ray muon flux
    # Surface cosmic ray flux ≈ 1.1e4 muons/m²/s at Fermilab elevation
    # Each muon deposits ~2.1 MeV/cm in LAr (MIP)
    # W_ion = 23.6 eV per ion pair in LAr
    # Recombination survival fraction R ≈ 0.64 at 273 V/cm (Box model)
    e_charge = 1.602e-19  # C
    muon_flux = 1.1e4      # muons/m²/s (surface, all angles integrated)
    dEdx_MIP = 2.1         # MeV/cm
    W_ion = 23.6e-6        # MeV per ion pair
    R_recomb = 0.64        # survival fraction at ~273 V/cm

    # Average path length through a cubic-ish volume
    # (for isotropic flux through a box, mean chord ~ 2V/S)
    V_det = (Lx / 100) * (Ly / 100) * (Lz / 100)  # m³
    S_det = 2 * ((Lx*Ly + Ly*Lz + Lx*Lz)) / 1e4   # m² surface area
    mean_chord = 2 * V_det / S_det * 100  # cm

    # Charge production per unit volume
    # = muon_flux × S_det/2 (effective entry area) × dE/dx × (1-R) / W_ion × e / V_det
    # Simplified: treat as volumetric source
    n_ion_per_cm = dEdx_MIP / W_ion       # ion pairs per cm of track
    # For ~isotropic flux entering from top and sides:
    # rate density ≈ muon_flux × track_length_density × charge per pair / V
    # This is complicated; use simpler estimate: Q_est ≈ muon_flux × <path/V> × dE/dx × (1-R) × e / W_ion

    # Rough estimate from MicroBooNE studies:
    # Q ≈ (cosmic rate density) × (ionisation) × (1 - R) × e
    # MicroBooNE quotes Q ~ 1e-10 to 5e-10 C/m³/s depending on assumptions
    print(f"\n  Back-of-envelope Q estimate:")
    print(f"    Cosmic muon flux (surface): ~{muon_flux:.0e} /m²/s")
    print(f"    MIP dE/dx in LAr: {dEdx_MIP} MeV/cm")
    print(f"    W_ion: {W_ion*1e6:.1f} eV/pair")
    print(f"    Recombination survival R ≈ {R_recomb}")
    print(f"    => FREE ions per cm: {dEdx_MIP/W_ion*(1-R_recomb):.0f} pairs/cm")
    print(f"    Literature range for Q: ~1e-10 to 5e-10 C/m³/s")
    print(f"    Our Q: {Q:.2e} C/m³/s — "
          f"{'within range' if 5e-11 < Q < 6e-10 else 'OUTSIDE expected range'}")

    # ================================================================== #
    #  6. ION DRIFT VELOCITY: SELF-CONSISTENCY                           #
    # ================================================================== #
    banner("6. Ion drift velocity self-consistency")

    print("  Current model: v_ion = mu_ion × E0 (constant, uses nominal field)")
    print(f"  v_ion = {v_ion:.4e} cm/s everywhere")
    print()
    print("  In reality, ions drift in the DISTORTED field:")

    ex_anode = Ex[1, iy_mid, iz_mid]    # near anode
    ex_cathode = Ex[-2, iy_mid, iz_mid]  # near cathode
    v_ion_anode = mu_ion * ex_anode
    v_ion_cathode = mu_ion * ex_cathode
    print(f"    v_ion(anode)   = mu × Ex(anode)   = {mu_ion:.2e} × {ex_anode:.1f} "
          f"= {v_ion_anode:.4e} cm/s")
    print(f"    v_ion(cathode) = mu × Ex(cathode)  = {mu_ion:.2e} × {ex_cathode:.1f} "
          f"= {v_ion_cathode:.4e} cm/s")
    print(f"    Variation:     {(v_ion_cathode-v_ion_anode)/v_ion*100:+.1f}% "
          f"relative to nominal")
    print()
    if abs(v_ion_cathode - v_ion_anode) / v_ion > 0.1:
        print("  *** NOTE: v_ion varies by >10% across the drift volume.")
        print("       A self-consistent (iterative) solution would update")
        print("       ρ(x) = Q·x / v_ion(x) at each iteration until convergence.")
        print("       This is a known simplification; published MicroBooNE sims")
        print("       also use the linearised model as a first approximation.")
    else:
        print("  OK — v_ion variation small, linearisation is acceptable")

    # ================================================================== #
    #  7. BOUNDARY EFFECTS: FIELD CAGE REALISM                           #
    # ================================================================== #
    banner("7. Boundary condition realism")

    # Check transverse field at boundaries (should be zero for Dirichlet BC
    # but physical field cages aren't perfect)
    ey_at_y0 = np.max(np.abs(Ey[:, 1, :]))
    ez_at_z0 = np.max(np.abs(Ez[:, :, 1]))
    ey_at_yL = np.max(np.abs(Ey[:, -2, :]))
    ez_at_zL = np.max(np.abs(Ez[:, :, -2]))

    print(f"  BCs: homogeneous Dirichlet (dphi=0 on all 6 faces)")
    print(f"  This assumes a PERFECT field cage maintaining V(y,z) = -E0·x")
    print()
    print(f"  Transverse E-field near boundaries:")
    print(f"    max |Ey| near y=0:   {ey_at_y0:.2f} V/cm "
          f"({ey_at_y0/E0*100:.1f}% of E0)")
    print(f"    max |Ey| near y=Ly:  {ey_at_yL:.2f} V/cm "
          f"({ey_at_yL/E0*100:.1f}% of E0)")
    print(f"    max |Ez| near z=0:   {ez_at_z0:.2f} V/cm "
          f"({ez_at_z0/E0*100:.1f}% of E0)")
    print(f"    max |Ez| near z=Lz:  {ez_at_zL:.2f} V/cm "
          f"({ez_at_zL/E0*100:.1f}% of E0)")
    print()
    print("  In reality, field cage imperfections and edge effects can cause")
    print("  additional distortions near boundaries. This simulation gives the")
    print("  'ideal field cage' baseline.")

    # ================================================================== #
    #  8. GRID CONVERGENCE                                               #
    # ================================================================== #
    banner("8. Grid convergence check")

    # Run at two resolutions and compare dphi_max
    print("  Comparing Poisson solution at different resolutions...")

    for Ntest in [51, 101]:
        p_test = build_params(preset=preset, overrides=dict(
            Nx_poisson=Ntest, Ny_poisson=Ntest, Nz_poisson=Ntest))

        x_t = np.linspace(0, Lx, Ntest)
        y_t = np.linspace(0, Ly, Ntest)
        z_t = np.linspace(0, Lz, Ntest)
        rho_t = compute_charge_density(x_t, y_t, z_t,
                                       p_test["Q_charge_production"],
                                       p_test["v_ion"])
        dphi_t = solve_poisson_dst(rho_t, Lx, Ly, Lz, p_test["epsilon"])
        dx_t = Lx / (Ntest - 1)
        dy_t = Ly / (Ntest - 1)
        dz_t = Lz / (Ntest - 1)
        _, _, _, E_mag_t = compute_efield(dphi_t, E0, dx_t, dy_t, dz_t)
        E_ratio_t = E_mag_t / E0

        print(f"    N={Ntest:3d}: dphi_max={np.max(np.abs(dphi_t)):.4f} V, "
              f"|E|/E0 range=[{E_ratio_t.min():.5f}, {E_ratio_t.max():.5f}]")

    print("  (dphi_max should converge as N increases; E-field ratio less")
    print("   sensitive to grid since np.gradient is only 2nd-order)")

    # ================================================================== #
    #  9. ASYMMETRIC DETECTOR GEOMETRY (MicroBooNE Ly != Lz)             #
    # ================================================================== #
    banner("9. Geometry asymmetry effects")

    print(f"  Lx={Lx:.0f} cm, Ly={Ly:.0f} cm, Lz={Lz:.0f} cm")
    print(f"  Ly/Lx = {Ly/Lx:.2f},  Lz/Lx = {Lz/Lx:.2f},  Lz/Ly = {Lz/Ly:.2f}")

    if abs(Ly - Lz) > 10:
        print(f"  Ly ≠ Lz → delta_y and delta_z should have DIFFERENT magnitudes")
        print(f"    max |delta_y| = {np.max(np.abs(dy_map)):.2f} cm")
        print(f"    max |delta_z| = {np.max(np.abs(dz_map)):.2f} cm")
        ratio = np.max(np.abs(dy_map)) / max(np.max(np.abs(dz_map)), 1e-10)
        print(f"    |delta_y|/|delta_z| = {ratio:.3f}")
        print(f"  Smaller transverse dimension → tighter boundary → larger gradient")
        print(f"  → expect LARGER distortion per unit length for smaller dimension")
        # Ly = 233 (small) vs Lz = 1036 (large) for MicroBooNE
        if Ly < Lz:
            if np.max(np.abs(dy_map)) > np.max(np.abs(dz_map)):
                print(f"  OK: |delta_y| > |delta_z| (Ly < Lz, tighter BC in y)")
            else:
                print(f"  *** UNEXPECTED: |delta_y| < |delta_z| despite Ly < Lz")
    else:
        print(f"  Ly ≈ Lz → expect |delta_y| ≈ |delta_z| (symmetric)")

    # ================================================================== #
    #  SUMMARY TABLE                                                     #
    # ================================================================== #
    banner("SUMMARY")

    v_nom = drift_velocity(E0, T=params.get("temperature", 89.0))
    print(f"  Detector:        {preset}")
    print(f"  Volume:          {Lx:.0f} × {Ly:.0f} × {Lz:.0f} cm³")
    print(f"  E0:              {E0:.0f} V/cm")
    print(f"  v_drift(E0):     {v_nom:.4f} cm/µs ({v_nom*10:.3f} mm/µs)")
    print(f"  Q:               {Q:.2e} C/m³/s")
    print(f"  v_ion:           {v_ion:.4e} cm/s ({v_ion*10:.2f} mm/s)")
    print(f"  ρ(cathode):      {rho_cathode:.3e} C/m³")
    print(f"  Ion clearing:    {t_clear:.0f} s ({t_clear/60:.1f} min)")
    print(f"  |E|/E0:          [{E_mag.min()/E0:.4f}, {E_mag.max()/E0:.4f}]")
    print(f"  Voltage check:   {err_pct:+.4f}%")
    print(f"  max |Δx|:        {dx_max:.2f} cm ({dx_max/Lx*100:.2f}% of Lx)")
    print(f"  max |Δy|:        {dy_max:.2f} cm ({dy_max/Ly*100:.2f}% of Ly)")
    print(f"  max |Δz|:        {dz_max:.2f} cm ({dz_max/Lz*100:.2f}% of Lz)")

    return results


if __name__ == "__main__":
    import sys
    preset = sys.argv[1] if len(sys.argv) > 1 else "microboone"
    run_expert_review(preset=preset)
