"""Light-only simulation: extract per-segment Q, L, position, t0, ancestor_id."""

import sys
import time
import numpy as np
import h5py
import jax
from tools.geometry import generate_detector
from tools.simulation import DetectorSimulator
from tools.loader import load_event

data_path = sys.argv[1] if len(sys.argv) > 1 else 'out.h5'

detector_config = generate_detector('config/cubic_wireplane_config.yaml')
sim = DetectorSimulator(
    detector_config,
    total_pad=400_000,
    response_chunk_size=50_000,
    include_track_hits=False,
)
cfg = sim.config

# Warm up JIT
deposits = load_event(data_path, cfg, event_idx=0)
filled = sim.process_event_light(deposits)
jax.block_until_ready(filled.volumes[0].charge)

# Count events in file
with h5py.File(data_path, 'r') as f:
    n_events = f['event/geant4'].shape[0]

print(f"\n{'Event':>5} {'Segs':>8} {'Load':>8} {'Sim':>8} {'Photons':>14} {'Charge':>14} {'Q/(Q+L)':>8}")
print("-" * 75)

load_times = []
sim_times = []

all_events = {}

for event_idx in range(min(n_events, 50)):
    t0 = time.time()
    deposits = load_event(data_path, cfg, event_idx=event_idx)
    t_load = (time.time() - t0) * 1000

    t0 = time.time()
    filled = sim.process_event_light(deposits)
    jax.block_until_ready(filled.volumes[0].charge)
    t_sim = (time.time() - t0) * 1000

    load_times.append(t_load)
    sim_times.append(t_sim)

    total_q = 0.0
    total_l = 0.0
    total_n = 0
    event_volumes = {}

    for v in range(cfg.n_volumes):
        vol = filled.volumes[v]
        n = vol.n_actual
        if n == 0:
            continue

        charge = np.asarray(vol.charge[:n])
        photons = np.asarray(vol.photons[:n])
        positions = np.asarray(vol.positions_mm[:n])
        t0_us = np.asarray(vol.t0_us[:n])
        ancestor_ids = np.asarray(vol.ancestor_track_ids[:n])

        total_q += float(charge.sum())
        total_l += float(photons.sum())
        total_n += n

        event_volumes[v] = {
            'charge': charge,
            'photons': photons,
            'positions_mm': positions,
            't0_us': t0_us,
            'ancestor_track_ids': ancestor_ids,
        }

    all_events[event_idx] = event_volumes
    q_frac = total_q / (total_q + total_l) if (total_q + total_l) > 0 else 0

    print(f"{event_idx:>5} {total_n:>8,} {t_load:>7.1f}ms {t_sim:>7.1f}ms "
          f"{total_l:>14,.0f} {total_q:>14,.0f} {q_frac:>8.3f}")

print("-" * 75)
print(f"{'Mean':>14} {np.mean(load_times):>7.1f}ms {np.mean(sim_times):>7.1f}ms")
print(f"{'Std':>14} {np.std(load_times):>7.1f}ms {np.std(sim_times):>7.1f}ms")
