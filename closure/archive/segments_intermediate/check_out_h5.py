"""Check out.h5 structure and event sizes."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np
from tools.loader import load_particle_step_data

for ev in range(5):
    try:
        d = load_particle_step_data('out.h5', event_idx=ev, verbose=False)
        n = d.positions_mm.shape[0]
        dx = np.asarray(d.dx)
        de = np.asarray(d.de)
        pos = np.asarray(d.positions_mm)
        print(f'Event {ev}: {n:,} segs, total_dE={de.sum():.1f} MeV, '
              f'dx mean={dx.mean():.4f} median={np.median(dx):.4f} max={dx.max():.4f} mm, '
              f'x=[{pos[:,0].min():.0f},{pos[:,0].max():.0f}]')
    except Exception as e:
        print(f'Event {ev}: {e}')
        break
