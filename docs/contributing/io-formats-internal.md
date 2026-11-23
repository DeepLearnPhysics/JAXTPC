# Internal I/O formats

Event data has one HDF5 format: the production pipeline
(`production/save.py` writers, `production/load.py` readers; `sensor` / `step` /
`hits` [+ `labl`] files, any number of volumes, wire or pixel). The writers take an
open `h5py.File`, so they also work for saving a few events from a notebook. See
[Production data formats](../production/data-formats.md).

For in-memory conversion use `sim.to_sparse(...)` / `sim.to_dense(...)`
(`tools/output.py`); `to_dense` accepts every format `to_sparse` returns.

The former single-file `tools.utils.save_event` / `load_event` / `list_events`
were removed: they expected keys no current output produces and only supported
the 2-volume U/V/Y detector.

## `save_sce_data` / `load_sce_data` — SCE map I/O

`tools/utils.py` carries a small per-volume space-charge-effect (SCE) map format:

```python
from tools.utils import save_sce_data, load_sce_data

save_sce_data('sce.h5', volume_data, metadata={...})
volume_data = load_sce_data('sce.h5')   # list of per-volume dicts
```

Each volume is stored under `volume_{i}/` with `efield_map` and
`drift_correction_map` (both `(Nx, Ny, Nz, 3)` float32), plus `origin_cm` and
`spacing_cm`. Optional `metadata` is written as file-level attributes. Any number of
volumes is supported.

`save_sce_data` / `load_sce_data` are used by the test suite
(`tests/test_efield_distortions.py`) and are the intended interchange format for
the planned SCE-map port. See [Space charge effects](../physics/sce.md).

