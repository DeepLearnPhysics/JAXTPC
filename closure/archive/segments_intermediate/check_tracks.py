"""Check track structure in out.h5 event 2."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np
from tools.loader import load_particle_step_data

d = load_particle_step_data('out.h5', event_idx=2, verbose=False)
track_ids = np.asarray(d.track_ids)
pos = np.asarray(d.positions_mm)
de = np.asarray(d.de)
n = len(de)

unique_tracks, counts = np.unique(track_ids, return_counts=True)
n_tracks = len(unique_tracks)

print(f"Event 2: {n:,} segments, {n_tracks:,} tracks")
print(f"\nTrack size distribution:")
print(f"  min:    {counts.min()}")
print(f"  median: {int(np.median(counts))}")
print(f"  mean:   {counts.mean():.1f}")
print(f"  max:    {counts.max()}")
for p in [10, 25, 50, 75, 90, 95, 99]:
    print(f"  {p}th:   {int(np.percentile(counts, p))}")

print(f"\nTrack spatial extent (max-min per track):")
extents = []
for tid in unique_tracks[:500]:  # sample first 500 tracks
    mask = track_ids == tid
    if mask.sum() < 2:
        continue
    t_pos = pos[mask]
    extent = t_pos.max(axis=0) - t_pos.min(axis=0)
    extents.append(np.linalg.norm(extent))
extents = np.array(extents)
print(f"  Sampled {len(extents)} tracks with ≥2 segments")
print(f"  min extent:    {extents.min():.1f} mm")
print(f"  median extent: {np.median(extents):.1f} mm")
print(f"  mean extent:   {extents.mean():.1f} mm")
print(f"  max extent:    {extents.max():.1f} mm")
for p in [10, 25, 50, 75, 90, 95, 99]:
    print(f"  {p}th: {np.percentile(extents, p):.1f} mm")

print(f"\nTotal dE by track size:")
big = counts >= 100
med = (counts >= 10) & (counts < 100)
small = counts < 10
print(f"  Large (≥100 segs): {big.sum()} tracks, {sum(counts[big]):,} segs, "
      f"{sum(de[np.isin(track_ids, unique_tracks[big])]):.1f} MeV")
print(f"  Medium (10-99):    {med.sum()} tracks, {sum(counts[med]):,} segs, "
      f"{sum(de[np.isin(track_ids, unique_tracks[med])]):.1f} MeV")
print(f"  Small (<10):       {small.sum()} tracks, {sum(counts[small]):,} segs, "
      f"{sum(de[np.isin(track_ids, unique_tracks[small])]):.1f} MeV")

print(f"\n50mm jitter context:")
print(f"  Wire pitch: 3mm → 50mm = {50/3:.0f} wire pitches")
print(f"  Median track extent: {np.median(extents):.0f}mm")
print(f"  50mm jitter / median extent: {50/np.median(extents)*100:.0f}%")
