# Readout

Wire vs pixel readout.

| Notebook | What it does |
|---|---|
| `pixel_simulation.ipynb` | Single-event pixel-readout sim (full 1000×1000): projections / anode / waveforms |

The **ENC (wire) vs ADC (pixel) units convention** is documented in
[`docs/physics/units.md`](../../docs/physics/units.md) and noted inline in the
wire/pixel sim notebooks — it does not warrant a separate notebook (there is no
computation to walk through, just a units table). The wire path lives in
[`getting_started/wire_simulation.ipynb`](../getting_started/wire_simulation.ipynb);
this folder is where pixel-specific and any future wire-vs-pixel *comparison*
notebooks belong if a concrete comparison (beyond units) is wanted.
