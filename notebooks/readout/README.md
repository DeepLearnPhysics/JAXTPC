# Readout

Wire vs pixel readout, and the units convention that trips people up.

| Notebook | What it does |
|---|---|
| `pixel_simulation.ipynb` | Single-event pixel-readout sim (full 1000×1000), projections / anode / waveforms |

**Planned**
- `wire_vs_pixel_units.ipynb` — the two readouts side-by-side, and the
  **ENC vs ADC** convention made concrete

> **Units convention (the footgun):** wire hits are in **ENC** (electrons),
> pixel hits are in **ADC**. The thresholds `inter_thresh` / `threshold_adc` /
> `corr_threshold` therefore mean different things per readout. See
> `docs/physics/units.md`.
