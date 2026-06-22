# Reconstruction — closure methods

Reconstruction in JAXTPC **is** the closure methods in [`closure/`](../../closure):
differentiable optimization of an event's parameters through the simulator until
the simulated readout matches the observed (truth) readout. They all share the
same machinery — `DetectorSimulator(differentiable=True).forward_segments(...)`
as the forward, the **Sobolev geomean-log1p loss** (`tools/losses.py`,
`sobolev_loss_geomean_log1p` + `make_sobolev_weight`), and Adam — differing in
*what* is optimized.

| Notebook | Based on | Optimizes | 
|---|---|---|
| `segments_closure.ipynb` | `closure/segments/run.py` | an event as **N point charges** `[x, y, z, dE]`, with Adam + **MCMC relocation** of dead segments (3DGS-MCMC style) |
| `muon_closure.ipynb` | `closure/muon/run.py` | a muon track from initial guesses (track-surface parameterization) |

Each notebook will be a runnable walkthrough of the corresponding `closure/`
script on a small event: build truth signals → build the differentiable forward
→ Sobolev loss → optimize → show the loss curve and truth-vs-reconstruction
overlay. A minimal segments closure has been verified end-to-end (gradient +
40-step Adam, loss 1.04 → 0.56).

> The **MCS closure** (`closure/mcs/`) is intentionally *not* surfaced as a
> notebook — it is research code (see `closure/mcs/FINDINGS_MCS.md`), kept in
> `closure/` only.

> The full closure scripts read a real edepsim HDF5 (`load_particle_step_data`).
> The notebooks use a synthetic truth event so they run with no external data;
> the `closure/` scripts remain the reference for full-scale runs.

**Not in scope** (per project direction): learned/NN reconstruction and
wire-crossing "space points" — these are not the closure standard.
