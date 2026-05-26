"""Detailed track size histogram."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np
from tools.loader import load_particle_step_data

d = load_particle_step_data('out.h5', event_idx=2, verbose=False)
track_ids = np.asarray(d.track_ids)
de = np.asarray(d.de)

unique_tracks, counts = np.unique(track_ids, return_counts=True)

print(f"{'N segs':>8} {'N tracks':>10} {'Total segs':>12} {'Total dE (MeV)':>15} {'% of dE':>8}")
print("-" * 58)

for n in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
    mask = counts == n
    n_tracks = mask.sum()
    n_segs = n * n_tracks
    track_mask = np.isin(track_ids, unique_tracks[mask])
    total_de = de[track_mask].sum()
    pct = total_de / de.sum() * 100
    print(f"{n:>8} {n_tracks:>10} {n_segs:>12} {total_de:>15.1f} {pct:>8.1f}%")

for lo, hi in [(11, 20), (21, 50), (51, 100), (101, 500), (501, 1000), (1001, 5000), (5001, 100000)]:
    mask = (counts >= lo) & (counts <= hi)
    n_tracks = mask.sum()
    n_segs = counts[mask].sum()
    track_mask = np.isin(track_ids, unique_tracks[mask])
    total_de = de[track_mask].sum()
    pct = total_de / de.sum() * 100
    print(f"{lo}-{hi:>5} {n_tracks:>10} {n_segs:>12} {total_de:>15.1f} {pct:>8.1f}%")

print(f"\n{'Total':>8} {len(unique_tracks):>10} {len(de):>12} {de.sum():>15.1f} {'100.0':>8}%")
