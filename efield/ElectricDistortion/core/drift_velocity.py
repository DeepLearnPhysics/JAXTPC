"""Electron drift velocity parameterisation in liquid argon.

Uses the Walkowiak parameterisation (NIM A 449, 2000, 288-294) with the
parameter set used by LArSoft / ICARUS.  Input field in V/cm, output
velocity in cm/us.
"""

import numpy as np


# Walkowiak / LArSoft parameters
_P1 = -0.04640
_P2 = 0.01712
_P3 = 1.88125
_P4 = 0.99408
_P5 = 0.01172
_P6 = 4.20214
_T0 = 105.749  # reference temperature (K)


def drift_velocity(E_mag, T=89.0):
    """Electron drift velocity in liquid argon.

    Parameters
    ----------
    E_mag : float or ndarray
        Electric field magnitude (V/cm).  Must be > 0.
    T : float
        LAr temperature (K).

    Returns
    -------
    v_d : same shape as E_mag
        Drift velocity in cm/us.
    """
    E_mag = np.asarray(E_mag, dtype=float)
    tshift = T - _T0

    E_kV = E_mag / 1000.0  # V/cm -> kV/cm

    # Guard against E_kV == 0 (log would blow up)
    safe_E = np.where(E_kV > 0, E_kV, 1.0)

    vd_mm_us = (
        (_P1 * tshift + 1.0)
        * (_P3 * safe_E * np.log(1.0 + _P4 / safe_E) + _P5 * safe_E ** _P6)
        + _P2 * tshift
    )

    # Result is in mm/us; convert to cm/us
    vd = vd_mm_us / 10.0

    # Zero out where field was zero
    vd = np.where(E_kV > 0, vd, 0.0)

    return float(vd) if vd.ndim == 0 else vd
