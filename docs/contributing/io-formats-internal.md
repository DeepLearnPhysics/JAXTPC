# Internal I/O: the standalone single-file format

JAXTPC has **two** HDF5 I/O paths. This page documents the smaller one —
`tools/utils.py` — and, crucially, when *not* to use it.

| | `tools/utils.py` | `production/save.py` + `load.py` |
|---|---|---|
| Files per run | **one** (events appended as groups) | **multiple** per batch (`sensor` / `step` / `hits` [+ `labl`]) |
| Audience | interactive, one-off, notebooks | batch production, downstream ML |
| Geometry support | **2-volume U/V/Y only** | any N volumes, wire or pixel |
| Encoding | plain gzip datasets | delta + CSR, configurable codecs |
| Truth machinery | flat per-track hits | group→track correspondence, `qs_fractions`, per-volume `group_to_track` |

!!! warning "Do not use `tools/utils.py` for production"
    `tools/utils.py` is an interactive single-file *convenience* format. It is
    **not** the production output format and is not interchangeable with it.
    For batch runs, HDF5 schemas consumed by downstream tooling, or anything
    beyond the default 2-volume wire detector, use the production pipeline
    (`production/run_batch.py` → `production/save.py` / `load.py`).

## `save_event` / `load_event` — single-file event I/O

`tools/utils.py` saves both output paths (response + track hits) for an event
into **one** HDF5 file, appending events as `/event_{idx}/` groups, with the
detector config written once at the root:

```python
from tools.utils import save_event, load_event, list_events

# Save (creates the file, or appends this event to an existing one)
save_event('run.h5', event_idx, sparse_output, track_hits, detector_config)

# Load one event back
sparse_output, track_hits, config = load_event('run.h5', event_idx)

list_events('run.h5')   # -> sorted event indices in the file
```

The file layout is:

```text
/config/
    num_wires_actual   (2, 3) int32
    min_wire_indices   (2, 3) int32
    attrs: num_time_steps, electrons_per_adc
/event_{idx}/
    response/{plane}/   indices (N,2), values (N,), signal (N_s,)
    hits/{plane}/       hits_by_track (H,3), track_boundaries (T,), track_ids (T,)
```

This is a convenience wrapper for interactive work: quick single-file
save/reload of a handful of events without standing up the batch pipeline or its
threaded save workers, CSR encoding, and multi-file layout.

### The 2-volume U/V/Y limitation

Both `save_event` and `load_event` key planes through a hardcoded lookup,
`_PLANE_NAMES`, that only knows the standard **2-volume, 3-plane (U/V/Y)**
detector:

```python
_PLANE_NAMES = {
    (0, 0): 'vol0_U', (0, 1): 'vol0_V', (0, 2): 'vol0_Y',
    (1, 0): 'vol1_U', (1, 1): 'vol1_V', (1, 2): 'vol1_Y',
}
```

!!! warning "`save_event` KeyErrors on any non-U/V/Y geometry"
    `save_event` looks up `_PLANE_NAMES[(vol_idx, plane_idx)]` for every plane
    in its input. Any `(vol_idx, plane_idx)` outside the six pairs above — a
    third volume, a pixel volume, a detector with more or fewer planes — raises
    `KeyError`. `load_event` iterates the fixed name→key map, so it silently
    reads back only U/V/Y planes even if others were somehow written. There is a
    general `get_plane_name(vol_idx, plane_idx, plane_names)` helper in the same
    module, but `save_event`/`load_event` do **not** use it — they are hardwired
    to the 2-volume layout. For any other geometry, use the production pipeline.

## `save_sce_data` / `load_sce_data` — SCE map I/O

The same module carries a small per-volume space-charge-effect (SCE) map format,
independent of the event I/O above:

```python
from tools.utils import save_sce_data, load_sce_data

save_sce_data('sce.h5', volume_data, metadata={...})
volume_data = load_sce_data('sce.h5')   # list of per-volume dicts
```

Each volume is stored under `volume_{i}/` with `efield_map` and
`drift_correction_map` (both `(Nx, Ny, Nz, 3)` float32), plus `origin_cm` and
`spacing_cm`. Optional `metadata` is written as file-level attributes. This
format is not tied to the 2-volume limitation of the event I/O — it stores an
arbitrary list of volumes.

`save_sce_data` / `load_sce_data` are used by the test suite
(`tests/test_efield_distortions.py`) and are the intended interchange format for
the planned SCE-map port. See [Space charge effects](../physics/sce.md).

## When to use which

- **`tools/utils.py` event I/O** — quick interactive save/reload of a few events
  from the **default 2-volume wire detector**, in a notebook or script, when you
  don't want the batch machinery. Never for production or non-U/V/Y geometry.
- **`tools/utils.py` SCE I/O** — reading/writing SCE map files (tests, SCE port).
- **`production/save.py` + `load.py`** — everything else: batch runs, the
  `sensor`/`step`/`hits` schemas that downstream ML tooling reads, any N-volume
  or pixel geometry, and compressed/CSR-encoded output. See
  [Production data formats](../production/data-formats.md).
