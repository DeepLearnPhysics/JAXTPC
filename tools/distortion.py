"""Unified drift-field distortion interface.

A *distortion* of the drift field (space charge is one cause, field-cage defects /
HV non-uniformity others) is represented by ONE pluggable function

    delta_fn(field_params, p) -> Δ      # displacement (3,) cm, local frame, anode-BC

and ONE generic ``apply_distortion`` that derives the E-field from Δ (jvp + Walkowiak
inversion) and assembles the per-deposit outputs. The representation (``siren`` /
``poly`` / ``grid``) is chosen at construction; ``field_params`` lives in
``SimParams`` (differentiable for recovery, frozen for production). ``bake`` evaluates
any rep onto a Δ-grid (the recover→produce bridge).

E is never stored or fit — it is a consequence of Δ. Only recombination consumes E,
and derived-E matches the true Poisson E to ~0.01% in charge (<< noise), so a single
Δ-only contract is sufficient. See ``SCE_FIELD_DESIGN.md``.
"""
import jax
import jax.numpy as jnp
from jax.scipy.ndimage import map_coordinates

from tools.config import DistortionOutputs
from tools.sce_siren import efield_from_dDdx, siren_forward, _to_norm

_X_HAT = jnp.array([1.0, 0.0, 0.0])


# --------------------------------------------------------------------------
# Representations: single-point delta_fn(field_params, p:(3,)) -> Δ:(3,) cm.
# Each evaluates in the normalized local frame and enforces the anode BC
# INTERNALLY (siren/poly via the (x_norm+1) factor; grid carries Δ=0 in its data).
# --------------------------------------------------------------------------
def siren_delta(fp, p):
    return siren_forward(fp['weights'], fp['biases'], fp['omega_0'],
                         _to_norm(p, fp['norm_offsets'], fp['norm_scales']))


def make_poly_delta(exps):
    """Build a poly delta_fn capturing the static integer exponents ``exps``."""
    def poly_delta(fp, p):
        xn = _to_norm(p, fp['norm_offsets'], fp['norm_scales'])
        mon = jnp.stack([xn[0] ** a * xn[1] ** b * xn[2] ** c for (a, b, c) in exps])
        return (mon @ fp['coeffs']) * (xn[0] + 1.0)
    return poly_delta


def grid_delta(fp, p):
    c = (p - fp['origin']) / fp['spacing']          # fractional grid coords (3,)
    cc = c[:, None]
    return jnp.stack([map_coordinates(fp['grid'][..., k], cc, order=1, mode='nearest')[0]
                      for k in range(3)])


# Valid distortion types. 'none' is the explicit no-distortion option — a
# first-class choice (config/file may set type='none'), not just an absent file.
REPS = ('none', 'siren', 'poly', 'grid')


def make_delta_fn(rep, exps=None):
    """Resolve a rep tag to its delta_fn (static choice).

    Returns None for 'none' (the no-distortion type) — the caller uses
    ``nominal_outputs`` instead of ``apply_distortion``.
    """
    if rep == 'none':
        return None
    if rep == 'siren':
        return siren_delta
    if rep == 'poly':
        return make_poly_delta(exps)
    if rep == 'grid':
        return grid_delta
    raise ValueError(f"unknown distortion type {rep!r} (valid: {REPS})")


def nominal_outputs(positions_cm):
    """No-distortion (type='none') outputs: unit field, zero drift corrections."""
    n = positions_cm.shape[0]
    return DistortionOutputs(
        efield_correction=jnp.broadcast_to(jnp.array([1.0, 0.0, 0.0]), (n, 3)),
        drift_time_corr_us=jnp.zeros(n),
        drift_yz_corr_cm=jnp.zeros((n, 2)))


# --------------------------------------------------------------------------
# The ONE shared tail: Δ -> (E, drift-time, transverse) -> outputs.
# --------------------------------------------------------------------------
def apply_distortion(delta_fn, fp, positions_cm, velocity_cm_us, shared):
    # sanitize padding before jvp/interp: NaN/±inf → 0 (a finite in-frame point).
    # Padding deposits are masked downstream; they only need to stay finite so the
    # forward (and the reverse pass) don't get NaN-poisoned via compute_phi_drift.
    xyz = jnp.nan_to_num(positions_cm, nan=0.0, posinf=0.0, neginf=0.0)
    delta = jax.vmap(lambda p: delta_fn(fp, p))(xyz)
    dDdx = jax.vmap(lambda p: jax.jvp(lambda q: delta_fn(fp, q), (p,), (_X_HAT,))[1])(xyz)
    E = efield_from_dDdx(dDdx, fp['E0'], fp['v0'], shared['v_table'], shared['E_table'])
    E = E.at[:, 0].multiply(-fp['drift_direction'])  # SINGLE Ex flip to global frame
    E_norm = E / shared['nominal_field']
    x0 = xyz[:, 0]
    t_drift = (x0 + delta[:, 0]) / fp['v0']
    delta_t = t_drift - x0 / velocity_cm_us
    return DistortionOutputs(efield_correction=E_norm,
                      drift_time_corr_us=delta_t,
                      drift_yz_corr_cm=delta[:, 1:3])


# --------------------------------------------------------------------------
# bake: evaluate any rep onto a Δ-grid -> grid field_params (recover→produce).
# Re-injects geometry metadata; does NOT re-apply the anode BC (it's in delta_fn).
# --------------------------------------------------------------------------
def bake(delta_fn, fp, origin, spacing, shape):
    axes = [origin[i] + spacing[i] * jnp.arange(shape[i]) for i in range(3)]
    pts = jnp.stack(jnp.meshgrid(*axes, indexing='ij'), -1).reshape(-1, 3)
    grid = jax.vmap(lambda p: delta_fn(fp, p))(pts).reshape(*shape, 3)
    return {'grid': grid, 'origin': jnp.asarray(origin, jnp.float32),
            'spacing': jnp.asarray(spacing, jnp.float32),
            'E0': fp['E0'], 'v0': fp['v0'], 'drift_direction': fp['drift_direction']}
