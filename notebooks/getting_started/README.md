# Getting Started

Your first JAXTPC simulation, end-to-end. Start here.

| Notebook | What it does |
|---|---|
| `00_quickstart.ipynb` | Minimal: synthetic event → wire sim → a couple of plots, ~30 lines |
| `wire_simulation.ipynb` | Full single-event wire walkthrough: config, run, sparse output, track-hit truth, 4 plots |
| `custom_detector.ipynb` | Author a new geometry end-to-end: modify a preset YAML, load it, run a synthetic event |

**Planned**
- `detector_config.ipynb` — anatomy of a detector YAML; multi-volume; visualize geometry

These run self-contained on a synthetic multi-track event (no external data
file). To simulate real data, replace `make_synthetic_event(...)` with
`load_event(path, cfg, event_idx=...)`.
