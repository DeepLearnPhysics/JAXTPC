"""Check: is total Q correct even though total dE is high?"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np
from tools.loader import load_particle_step_data

DX_MM = 0.3
DX_CM = DX_MM / 10.0
ALPHA = 0.93
BETA = 0.212
LAR_DENSITY = 1.396

from tools.geometry import generate_detector
detector_config = generate_detector('config/cubic_wireplane_config.yaml')
field_kVcm = detector_config['electric_field']['field_strength'] / 1000.0
B_EFF = BETA / LAR_DENSITY / field_kVcm

def de_to_Q(de, dx_cm):
    return (dx_cm / B_EFF) * np.log(ALPHA + B_EFF * de / dx_cm)

# Truth
deposit_data = load_particle_step_data('out.h5', event_idx=2, verbose=False)
truth_de = np.asarray(deposit_data.de)
n_truth = len(truth_de)
truth_Q = de_to_Q(truth_de, DX_CM)
total_truth_Q = truth_Q.sum()
total_truth_dE = truth_de.sum()

print(f"=== TRUTH ===")
print(f"  N segments: {n_truth:,}")
print(f"  Total dE: {total_truth_dE:.1f} MeV")
print(f"  Total Q:  {total_truth_Q:.1f}")
print(f"  Mean dE:  {truth_de.mean():.6f} MeV")
print(f"  Mean Q:   {truth_Q.mean():.6f}")

# Optimizer final state
data = np.load('closure_analysis_full/sweeps/run_out_50k.npz')
final_params = data['final_params']
sim_de = final_params[:, 3]
alive = sim_de > 0.012
n_alive = alive.sum()
n_dead = (~alive).sum()

sim_Q = de_to_Q(sim_de, DX_CM)
sim_Q_alive = sim_Q[alive]

total_sim_Q = sim_Q.sum()
total_sim_Q_alive = sim_Q_alive.sum()
total_sim_dE = sim_de.sum()
total_sim_dE_alive = sim_de[alive].sum()

print(f"\n=== OPTIMIZER (final) ===")
print(f"  N segments: {len(sim_de):,} ({n_alive:,} alive, {n_dead:,} dead)")
print(f"  Total dE: {total_sim_dE:.1f} MeV")
print(f"  Total Q:  {total_sim_Q:.1f}")
print(f"  Total Q (alive only): {total_sim_Q_alive:.1f}")
print(f"  Mean dE (alive): {sim_de[alive].mean():.6f} MeV")
print(f"  Mean Q (alive):  {sim_Q_alive.mean():.6f}")

print(f"\n=== RATIOS ===")
print(f"  dE ratio (total): {total_sim_dE / total_truth_dE:.3f}")
print(f"  Q ratio (total):  {total_sim_Q / total_truth_Q:.3f}")
print(f"  Q ratio (alive):  {total_sim_Q_alive / total_truth_Q:.3f}")

# What dE_ratio would give Q_ratio=1.0?
# If all alive segments had uniform dE:
# n_alive * Q(dE_uniform) = total_truth_Q
# Q(dE_uniform) = total_truth_Q / n_alive
Q_target_per = total_truth_Q / n_alive
# Invert: dE = (dx/B) * (exp(B*Q/dx) - alpha)
dE_target = (DX_CM / B_EFF) * (np.exp(B_EFF * Q_target_per / DX_CM) - ALPHA)
total_dE_target = n_alive * dE_target + n_dead * 0.012
dE_ratio_at_Q1 = total_dE_target / total_truth_dE

print(f"\n=== EXPECTED dE RATIO FOR Q_ratio=1.0 ===")
print(f"  If {n_alive:,} alive segments produce truth total Q:")
print(f"    Q per alive segment needed: {Q_target_per:.6f}")
print(f"    dE per alive segment needed: {dE_target:.6f} MeV")
print(f"    Total dE: {total_dE_target:.1f} MeV")
print(f"    dE ratio: {dE_ratio_at_Q1:.3f}")
print(f"  This is the EXPECTED dE ratio — not a bug!")

# Verify with per-percentile
print(f"\n=== dE vs Q DISTRIBUTION ===")
print(f"  {'Percentile':>12} {'dE (MeV)':>12} {'Q':>12} {'Q/dE':>8}")
for p in [50, 90, 95, 99, 99.9]:
    de_p = np.percentile(sim_de[alive], p)
    q_p = de_to_Q(de_p, DX_CM)
    print(f"  {p:>12.1f} {de_p:>12.6f} {q_p:>12.6f} {q_p/de_p:>8.3f}")
