# JAXTPC Notebooks

Hands-on, runnable notebooks organized by theme. Most are **self-contained** —
they generate a synthetic event inline, so they run with no external data (swap
`make_synthetic_event(...)` for `load_event(path, cfg, event_idx=...)` to use real
data). Notebooks resolve the repo root themselves, so they run from any folder
(or from the repo root).

Outputs are stripped on commit by the `nbstripout` filter — after cloning run
once: `pip install nbstripout && nbstripout --install --attributes .gitattributes`.

## Themes

| Folder | Theme | For |
|---|---|---|
| [`getting_started/`](getting_started/) | First simulation, end-to-end | new users |
| [`physics/`](physics/) | The response chain: recombination, drift, diffusion, kernels, SCE | physics analysts |
| [`readout/`](readout/) | Wire vs pixel readout and the ENC/ADC units convention | all |
| [`gradients/`](gradients/) | Differentiable path, gradient visualizations, optimization | ML / reco |
| [`reco/`](reco/) | Reconstruction demos and studies | ML / reco |
| [`calibration/`](calibration/) | Calibration workflows | analysts / production |
| [`production/`](production/) | Batch pipeline, viewing output, capacity profiling | internal production |

See [`docs/PLAN.md`](../docs/PLAN.md) for the full documentation & notebook
roadmap, and each folder's `README.md` for what belongs there.

## Running

```bash
# from the repo root (or any folder — the notebooks resolve the root)
jupyter lab notebooks/getting_started/00_quickstart.ipynb
```

Heavy notebooks (full-resolution pixel, GIF sweeps) note their hardware needs at
the top.
