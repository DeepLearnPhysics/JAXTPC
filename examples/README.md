# Examples: differentiable reconstruction with JAXTPC

Each example fits something through the differentiable simulator
(`DetectorSimulator.forward_segments`) against a simulated truth. Run them from
the repository root; outputs go to `examples/<name>/outputs/` (git-ignored) or to
`$JAXTPC_EXAMPLES_OUT`. A GPU is recommended.

| Example | What it fits | Command | Reference result (A100) |
|---|---|---|---|
| `muon/` | a muon track: vertex, direction, energy | `python -m examples.muon.run A --steps 1200` | start A converges to <0.3 mm, <0.05°, 1 MeV of 500 MeV (~80 s). Start B stalls in a local minimum. |
| `muon/detector_surfaces.py` | loss and AD-vs-finite-difference gradient vs drift velocity, lifetime, diffusion, recombination | `python -m examples.muon.detector_surfaces` | lifetime and diffusion: AD = FD to <0.5%. |
| `muon/event_display.py`, `muon/render_3d.py` | plots for the muon example | `python -m examples.muon.event_display` | |
| `segments/` | a whole event as N point charges `[x, y, z, dE]` (Adam + relocation) | `python -m examples.segments.run --data muon.h5 --n-seg 2000 --steps 2000 --mode full` | on `muon.h5`: mean distance to truth 0.8 mm (from 47 mm), charge ratio 1.000 (~6 min). |
| `segments/validate_convergence.py` | gradient direction and loss reduction check | `python -m examples.segments.validate_convergence --data muon.h5 --n-seg 1000` | ~89% of position gradients point toward truth. |
| `mcs/` | a scattering muon: globals, then per-segment scattering angles | `python -m examples.mcs.run --mode staged --steps 1050 --stage1-steps 250 --lr-phase2 0.003 --lr-scat 0.01 --decay 0.9995` | vertex 0.4 ± 0.2 mm, energy −6 ± 8 MeV over 5 seeds; net scattering angle recovered (corr. 0.98), individual 3 mm kinks not (corr. 0.24–0.37). |
| `mcs/validate_forward.py` | MCS forward model: geometry, CSDA, AD vs FD (no detector) | `python -m examples.mcs.validate_forward` | 14/14 checks pass. |

`FINDINGS_MCS.md` and `segments/FINDINGS_S1.md` record how the recipes were chosen.

Known limitation: `forward_segments` has no per-deposit start time (t0), so
`segments/validate_pipeline.py` (production vs differentiable consistency) fails on
events with nonzero t0 (loss 0.17 on `muon.h5`; exact with t0 = 0).
