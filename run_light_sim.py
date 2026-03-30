"""Benchmark light-only simulation across multiple events."""

import sys
import time
import numpy as np
import jax
from tools.geometry import generate_detector
from tools.simulation import DetectorSimulator
from tools.loader import load_particle_step_data

data_path = sys.argv[1] if len(sys.argv) > 1 else 'out.h5'

detector_config = generate_detector('config/cubic_wireplane_config.yaml')
sim = DetectorSimulator(
    detector_config,
    total_pad=200_000,
    response_chunk_size=50_000,
    include_track_hits=False,
)

# Warm up JIT
deposit_data, _ = load_particle_step_data(data_path, event_idx=0)
result = sim.process_event_light(deposit_data)
jax.block_until_ready(result['east'][0])

print(f"\n{'Event':>5} {'Segs':>8} {'Load':>8} {'Sim':>8} {'Photons':>14} {'Charge':>14} {'Q/(Q+L)':>8}")
print("-" * 75)

load_times = []
sim_times = []

for event_idx in range(20):
    t0 = time.time()
    deposit_data, _ = load_particle_step_data(data_path, event_idx=event_idx)
    t_load = (time.time() - t0) * 1000

    t0 = time.time()
    result = sim.process_event_light(deposit_data)
    jax.block_until_ready(result['east'][0])
    t_sim = (time.time() - t0) * 1000

    load_times.append(t_load)
    sim_times.append(t_sim)

    # Move to CPU for reporting only
    n_e, n_w = result['n_east'], result['n_west']
    Q_e, L_e, pos_e = [np.asarray(x) for x in result['east']]
    Q_w, L_w, pos_w = [np.asarray(x) for x in result['west']]

    total_q = Q_e[:n_e].sum() + Q_w[:n_w].sum()
    total_l = L_e[:n_e].sum() + L_w[:n_w].sum()
    q_frac = total_q / (total_q + total_l) if (total_q + total_l) > 0 else 0
    n = len(deposit_data.de)

    print(f"{event_idx:>5} {n:>8,} {t_load:>7.1f}ms {t_sim:>7.1f}ms "
          f"{total_l:>14,.0f} {total_q:>14,.0f} {q_frac:>8.3f}")

print("-" * 75)
print(f"{'Mean':>14} {np.mean(load_times):>7.1f}ms {np.mean(sim_times):>7.1f}ms")
print(f"{'Std':>14} {np.std(load_times):>7.1f}ms {np.std(sim_times):>7.1f}ms")
