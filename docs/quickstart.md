# Quickstart

The smallest end-to-end example: build a wire detector, generate a synthetic
event, run the simulation, and plot the readout. No external data needed.

For the runnable version, see
[`notebooks/getting_started/00_quickstart.ipynb`](https://github.com/DeepLearnPhysics/JAXTPC/blob/main/notebooks/getting_started/00_quickstart.ipynb).

```python
import jax, numpy as np, matplotlib.pyplot as plt
from tools.simulation import DetectorSimulator
from tools.geometry import generate_detector
from tools.loader import build_deposit_data
from tools.output import to_sparse
from tools.visualization import visualize_wire_signals

# 1. Build a dual-TPC wire detector
detector = generate_detector("config/cubic_wireplane_config.yaml")
sim = DetectorSimulator(detector, include_track_hits=True, include_digitize=True,
                        total_pad=50_000, response_chunk_size=10_000)
cfg = sim.config

# 2. Generate a synthetic event (a few straight MIP-like tracks)
def make_synthetic_event(seed=0, n_tracks=6, step_cm=0.4):
    rng = np.random.RandomState(seed)
    P, DE, DX, TID = [], [], [], []
    for t in range(n_tracks):
        start = rng.uniform(-180, 180, 3); d = rng.normal(size=3); d /= np.linalg.norm(d)
        n = int(rng.uniform(80, 250) / step_cm); s = np.arange(n) * step_cm
        pts = np.clip(start[None, :] + s[:, None] * d[None, :], -215.9, 215.9)
        P.append(pts); DE.append(np.full(n, rng.uniform(1.8, 2.6) * step_cm, np.float32))
        DX.append(np.full(n, step_cm, np.float32)); TID.append(np.full(n, t, np.int32))
    return (np.concatenate(P) * 10).astype(np.float32), np.concatenate(DE), np.concatenate(DX), np.concatenate(TID)

pos_mm, de, dx, track_ids = make_synthetic_event()
deposits = build_deposit_data(pos_mm, de, dx, cfg, track_ids=track_ids)

# 3. Run
sim.warm_up()
signals, track_hits_raw, deposits = sim.process_event(deposits, key=jax.random.PRNGKey(42))

# 4. Sparse output + plot
sparse = to_sparse(signals, cfg, threshold_adc=1200 / cfg.electrons_per_adc)
visualize_wire_signals(sparse, cfg, threshold_enc=1200, gamma=0.2, sparse=True)
plt.show()
```

To simulate **real data**, replace step 2 with:

```python
from tools.loader import load_event
deposits = load_event("events.h5", cfg, event_idx=0)
```

## Next steps

- [Reading guide](architecture/reading-guide.md) — how the code above actually works, function by function
- [`notebooks/getting_started/wire_simulation.ipynb`](https://github.com/DeepLearnPhysics/JAXTPC/blob/main/notebooks/getting_started/wire_simulation.ipynb) — full walkthrough with truth/track labels
- [`notebooks/readout/pixel_simulation.ipynb`](https://github.com/DeepLearnPhysics/JAXTPC/blob/main/notebooks/readout/pixel_simulation.ipynb) — pixel readout
- [Units (ENC vs ADC)](physics/units.md) and [Coordinates](concepts/coordinates.md) — the two conventions to know
