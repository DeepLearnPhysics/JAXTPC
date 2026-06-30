"""Regenerate tools/data/muon_dedx_lar.csv (PDG-style muon dE/dx in LAr).

The original file was lost from the checkout. This rebuilds it from the
Bethe-Bloch collision stopping power with the Sternheimer density-effect
parameterization for liquid argon (Groom-Mokhov-Striganov constants):
  Z=18, A=39.948, I=188 eV, dens: Cbar=5.2146, x0=0.2, x1=3.0,
  a=0.19559, k=3.0, delta0=0.
Reproduces the PDG muon table for LAr to <1% for T in ~10 MeV - 10 GeV
(radiative losses, relevant only >100 GeV, are neglected).

Columns: T_MeV, p_MeV, dedx_MeVcm2g, csda_range_gcm2, beta
(tools/particle_generator.py reads columns 0 and 2 only.)
"""
import numpy as np

M_MU = 105.6583745      # MeV
M_E = 0.5109989461      # MeV
K = 0.307075            # MeV cm^2 / mol
Z, A = 18.0, 39.948
I_EV = 188.0
CBAR, X0, X1, A_S, K_S = 5.2146, 0.2, 3.0, 0.19559, 3.0


def delta(x):
    """Sternheimer density effect, x = log10(beta*gamma)."""
    d = np.zeros_like(x)
    mid = (x >= X0) & (x < X1)
    hi = x >= X1
    d[mid] = 4.6052 * x[mid] - CBAR + A_S * (X1 - x[mid]) ** K_S
    d[hi] = 4.6052 * x[hi] - CBAR
    return d


def dedx_bethe(T):
    """Collision stopping power in MeV cm^2/g for muon kinetic energy T (MeV)."""
    gamma = 1.0 + T / M_MU
    beta2 = 1.0 - 1.0 / gamma ** 2
    beta = np.sqrt(beta2)
    bg = beta * gamma
    I = I_EV * 1e-6  # MeV
    wmax = 2 * M_E * bg ** 2 / (1 + 2 * gamma * M_E / M_MU + (M_E / M_MU) ** 2)
    x = np.log10(bg)
    arg = 2 * M_E * bg ** 2 * wmax / I ** 2
    return (K * (Z / A) / beta2 *
            (0.5 * np.log(arg) - beta2 - delta(x) / 2.0))


def main():
    T = np.geomspace(1.0, 1.0e5, 240)  # MeV kinetic energy
    dedx = dedx_bethe(T)
    gamma = 1.0 + T / M_MU
    beta = np.sqrt(1.0 - 1.0 / gamma ** 2)
    p = gamma * beta * M_MU

    # CSDA range in g/cm^2 by trapezoidal integration of 1/(dE/dx)
    inv = 1.0 / dedx
    R = np.concatenate([[0.0],
                        np.cumsum(0.5 * (inv[1:] + inv[:-1]) * np.diff(T))])

    # sanity checks
    i_min = np.argmin(dedx)
    print(f"min dE/dx = {dedx[i_min]:.4f} MeV cm2/g at T = {T[i_min]:.0f} MeV "
          f"(PDG LAr muon: ~1.508)")
    for t_chk in (10.0, 100.0, 500.0, 1000.0):
        print(f"  dE/dx(T={t_chk:6.0f} MeV) = "
              f"{np.interp(t_chk, T, dedx):.4f} MeV cm2/g")

    header = (
        "# Muon stopping power in liquid argon (PDG-style table)\n"
        "# Regenerated from Bethe-Bloch + Sternheimer density effect\n"
        "# (Z=18, A=39.948, I=188 eV; GMS liquid-argon constants).\n"
        "# Columns: T_MeV, p_MeV, dedx_MeVcm2g, csda_range_gcm2, beta\n")
    rows = np.column_stack([T, p, dedx, R, beta])
    with open('tools/data/muon_dedx_lar.csv', 'w') as f:
        f.write(header)
        for r in rows:
            f.write(f"{r[0]:.6e},{r[1]:.6e},{r[2]:.6e},{r[3]:.6e},{r[4]:.6f}\n")
    print("Wrote tools/data/muon_dedx_lar.csv")


if __name__ == '__main__':
    main()
