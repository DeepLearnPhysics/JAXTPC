# Closure Archive

Read-only reference for old/superseded experiments. These files have broken APIs
against the current `tools/` code and are kept only for historical reference.

## Contents

### `slides/`
Presentation figure scripts from `closure_analysis_muon/slides_plots/`. 8 of 13
scripts depend on scan-based muon generator functions that were never promoted
to `tools/`. The remaining 5 are standalone numpy/matplotlib scripts.

### `closure_analysis_original/`
The original closure prototype (Feb 24-26). Point deposits, Sliced Wasserstein
loss, SGLD/MCMC exploration. Fully superseded by `closure/muon/` and
`closure/segments/`.

### `gradient_test/`
Early gradient validation (Dec 21). Contains a forked copy of `tools/` from
that era. Proved differentiability works; superseded by the closure tests
that actually converge.

### `muon_intermediate/`
Superseded muon optimization scripts. The evolution:
- `muon_full_optimization.py` (sobolev_geomean, LR=0.015, 300 steps)
- `run_optimization_save.py` (same + NPZ saving)
- `run_multi_optimization.py` (4 inits, 1000 steps)
- `run_single_test.py` (test log1p variant)
- → `closure/muon/run.py` (terminal: log1p, LR=0.01, 600 steps, CLI)

Also includes `diff_muon_generator.py` (scan-based generators, canonical
CSDA parts now in `tools/particle_generator.py`) and completed studies
(angle optimization, energy segment study, SW diagnostics).

### `segments_intermediate/`
Superseded sweep runners. The evolution across 4 phases:
1. 10k segs on mpvmpr_20.h5, s=1.5 (run_sweep, run_best_steps, run_noise)
2. Scale to 20k segs (run_20k, run_s1_20k)
3. Switch to out.h5, 50k segs, s=1.0 (run_out_50k*, run_track_jitter*)
4. → `closure/segments/run_final.py` (terminal: best config, power-law checkpoints)
