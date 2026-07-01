# Examples

Runnable notebooks. The ones tagged **rendered here** execute at build time on a
synthetic event, so their plots appear in this site — nothing to install or
download. The rest are linked to GitHub to run yourself, either because they need
an input data file or an optional dependency.

## Rendered here (synthetic — plots included)

| Notebook | What it shows |
|---|---|
| [Quickstart](tutorials/00_quickstart.ipynb) | Smallest end-to-end run: synthetic event → wire sim → plots |
| [Physics walkthrough](tutorials/physics_walkthrough.ipynb) | One track, a plot at each pipeline stage (deposits → Q/L → drift/diffusion/response → signals) |
| [Custom detector](tutorials/custom_detector.ipynb) | Author a new geometry end-to-end by modifying a preset YAML |

## Run locally

Open these on GitHub and run them yourself. Each row notes what it needs beyond a
base install.

| Notebook | What it shows | Needs |
|---|---|---|
| [Response kernels](https://github.com/DeepLearnPhysics/JAXTPC/blob/main/notebooks/physics/response_kernels.ipynb) | Visualize the wire response kernels and the diffusion-convolved `DKernel` tables | run from the repo root (finds `config/`, `tools/responses/`) |
| [Optimization](https://github.com/DeepLearnPhysics/JAXTPC/blob/main/notebooks/gradients/optimization.ipynb) | Gradient-based fit through the differentiable path | `pip install optax` |
| [Segments closure](https://github.com/DeepLearnPhysics/JAXTPC/blob/main/notebooks/reco/segments_closure.ipynb) | Truth / track-hit correspondence and reconstruction closure | `optax` + the `closure/` research modules |
| [Wire simulation](https://github.com/DeepLearnPhysics/JAXTPC/blob/main/notebooks/getting_started/wire_simulation.ipynb) | Full single-event wire walkthrough with truth/track labels | an event `.h5` (or swap in `make_synthetic_event`) |
| [Pixel simulation](https://github.com/DeepLearnPhysics/JAXTPC/blob/main/notebooks/readout/pixel_simulation.ipynb) | Pixel readout: single-pass signal + truth | an event `.h5` |
| [View production](https://github.com/DeepLearnPhysics/JAXTPC/blob/main/notebooks/production/view_production.ipynb) | Load and visualize `run_batch.py` output | a production `sensor/step/hits` directory |

!!! tip "Running any notebook"
    ```bash
    pip install -e ".[dev]"
    jupyter lab notebooks/          # run from the repo root
    ```
    To feed real data into a synthetic notebook, replace the
    `make_synthetic_event(...)` call with `load_event(path, cfg, event_idx=...)`.
