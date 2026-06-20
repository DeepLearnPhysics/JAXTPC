# Production

Batch processing, output inspection, and capacity profiling — the internal
production workflow.

| Notebook | What it does |
|---|---|
| `view_production.ipynb` | Load and visualize a production HDF5 event (sensor / step / hits) — no simulation needed |

**Planned**
- `run_batch_walkthrough.ipynb` — run `production/run_batch.py` on a few events;
  the three (+1) HDF5 file types and their encodings
- `profiler.ipynb` — capacity sizing: `total_pad` / `maxg` / `max_keys`, what the
  profiler scans, and overflow behavior
- `output_formats.ipynb` — dense ↔ sparse ↔ bucketed; loading with
  `production/load.py`

`view_production.ipynb` requires an existing production output file (sensor/step/
hits). See `docs/production/` for batch, data-format, profiler, and slurm docs.
