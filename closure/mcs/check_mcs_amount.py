"""Sanity-check the MCS scattering amount in the truth generator.

Compares the reconstructed Highland theta0 + the cumsum-model net deflection
against the textbook single-thickness Highland expectation, and shows the
seed-to-seed spread (is seed 42 a low realization?).
"""
import jax
import jax.numpy as jnp
import numpy as np

from closure.mcs.mcs_physics import _ke_to_beta_p, highland_theta0, X0_LAR_CM, MUON_MASS_MEV
from closure.mcs.forward import generate_mcs_truth
from tools.particle_generator import (
    load_dedx_table_jax, _get_consistent_csda, _csda_energy_deposits,
)

log_T, dedx = load_dedx_table_jax()
N, step_mm = 2000, 0.5
step_cm = step_mm / 10.0
E0 = 500.0
L_cm = N * step_cm

print(f"muon mass={MUON_MASS_MEV} MeV, X0(LAr)={X0_LAR_CM} cm, "
      f"track L={L_cm} cm ({N} x {step_mm} mm)")

# --- per-step theta0 at 500 MeV ---
b, p, bp = _ke_to_beta_p(jnp.float32(E0))
th0_step = float(highland_theta0(bp, step_cm))
print(f"\n500 MeV: beta={float(b):.4f}, p={float(p):.1f} MeV, beta*p={float(bp):.1f} MeV")
print(f"  Highland theta0 per 0.5mm step  = {th0_step*1000:.3f} mrad  (per plane)")

# --- textbook single-thickness Highland over the full 100 cm (fixed E=500) ---
th_full = float(highland_theta0(bp, L_cm))
print(f"  single-thickness Highland (100cm, fixed 500 MeV) = {th_full*1000:.1f} mrad (plane RMS)")

# --- energy-correct prediction: sqrt(sum theta0(E_k)^2) over the track ---
R_cm, T_MeV = _get_consistent_csda(log_T, dedx)
R0 = float(jnp.interp(jnp.log(E0), jnp.log(T_MeV), R_cm))
idx = np.arange(N)
R_mid = np.maximum(R0 - (idx + 0.5) * step_cm, float(R_cm[0]))
E_mid = np.interp(R_mid, np.asarray(R_cm), np.asarray(T_MeV))
_, _, bp_mid = _ke_to_beta_p(jnp.asarray(E_mid))
th0_prof = np.asarray(highland_theta0(bp_mid, step_cm))
pred_plane_rms = np.sqrt(np.sum(th0_prof ** 2))
print(f"  E loss: {E0:.0f} -> {E_mid[-1]:.0f} MeV over the track")
print(f"  theta0/step: start {th0_prof[0]*1000:.2f} -> end {th0_prof[-1]*1000:.2f} mrad")
print(f"  predicted NET plane RMS = sqrt(sum theta0^2) = {pred_plane_rms*1000:.1f} mrad")
print(f"  predicted |net| (2D, =plane_rms*sqrt(2)) RMS = {pred_plane_rms*np.sqrt(2)*1000:.1f} mrad")

# --- realized net deflection across seeds ---
nets, segrms = [], []
for s in range(40):
    pos, de, dt1, dt2 = generate_mcs_truth(
        jnp.float32(E0), jnp.zeros(3, jnp.float32),
        jnp.float32(np.pi/4), jnp.float32(np.pi/6),
        step_mm, N, log_T, dedx, jax.random.PRNGKey(s))
    nets.append(float(jnp.sqrt(jnp.sum(dt1)**2 + jnp.sum(dt2)**2)) * 1000)
    segrms.append(float(jnp.sqrt(jnp.mean(dt1**2 + dt2**2))) * 1000)
nets = np.array(nets)
print(f"\nrealized per-seg RMS (combined 2 planes): {np.mean(segrms):.2f} mrad")
print(f"realized |net| deflection over 40 seeds: "
      f"mean={nets.mean():.1f}, median={np.median(nets):.1f}, "
      f"min={nets.min():.1f}, max={nets.max():.1f} mrad")
print(f"  seed 42 |net| = {nets[42] if len(nets)>42 else 'n/a'}", end='')
# seed 42 specifically
_, _, d1_42, d2_42 = generate_mcs_truth(
    jnp.float32(E0), jnp.zeros(3, jnp.float32),
    jnp.float32(np.pi/4), jnp.float32(np.pi/6),
    step_mm, N, log_T, dedx, jax.random.PRNGKey(42))
net42 = float(jnp.sqrt(jnp.sum(d1_42)**2 + jnp.sum(d2_42)**2)) * 1000
pct = 100 * np.mean(nets <= net42)
print(f"\n  seed 42 |net| = {net42:.1f} mrad  (~{pct:.0f}th percentile of the 40 seeds)")

# --- lateral displacement of the track from a straight line (seed 42) ---
pos42, _, _, _ = generate_mcs_truth(
    jnp.float32(E0), jnp.zeros(3, jnp.float32),
    jnp.float32(np.pi/4), jnp.float32(np.pi/6),
    step_mm, N, log_T, dedx, jax.random.PRNGKey(42))
pos42 = np.asarray(pos42)
straight = pos42[0] + (pos42[-1] - pos42[0]) * (np.arange(N)[:, None] / (N - 1))
lateral = np.linalg.norm(pos42 - straight, axis=1)
print(f"\nseed 42 lateral deviation from straight line: "
      f"max={lateral.max():.1f} mm, rms={np.sqrt(np.mean(lateral**2)):.1f} mm "
      f"(wire pitch = 3 mm)")
