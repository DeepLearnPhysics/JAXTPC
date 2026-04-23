# JAXTPC Production Pipeline

Batch simulation of particle events in a liquid argon TPC, producing structured HDF5 output for downstream analysis and ML training.

## Contents

```
production/
├── run_batch.py              # Main batch simulation script
├── save.py                   # HDF5 save functions (sensor/seg/inst encoding)
├── load.py                   # HDF5 load/decode functions
├── view_production.ipynb     # Visualize production output (no simulation needed)
└── README.md                 # This file
```

## Usage

From the project root:

```bash
# Basic run (10 events, 2 save workers, digitization on, noise/electronics off)
python3 production/run_batch.py --data events.h5 --events 10

# Using a production config (recommended — auto-sets total_pad, chunks, max_keys, thresholds)
python3 production/run_batch.py \
    --data events.h5 \
    --config config/cubic_wireplane_config.yaml \
    --production-config config/production_cubic_wireplane_config.yaml

# Full manual options
python3 production/run_batch.py \
    --data mpvmpr_20.h5 \
    --config config/cubic_wireplane_config.yaml \
    --dataset myrun \
    --outdir output/ \
    --events 1000 \
    --events-per-file 100 \
    --threshold-adc 2.0 \
    --workers 2 \
    --noise \
    --electronics \
    --no-track-hits
```

### Production Config

Instead of manually setting `--total-pad`, `--response-chunk`, `--max-keys`, etc., you can
generate an optimized config and load it with `--production-config`:

```bash
# Generate config (scans data, probes max_keys, finds optimal chunks)
python3 -m profiler.setup_production --data events.h5 --config config/cubic_wireplane_config.yaml

# Use it
python3 production/run_batch.py --data events.h5 \
    --config config/cubic_wireplane_config.yaml \
    --production-config config/production_cubic_wireplane_config.yaml
```

The config file stores performance and quality knobs:

```yaml
# config/production_cubic_wireplane_config.yaml
detector_config: config/cubic_wireplane_config.yaml
total_pad: 300000
response_chunk: 50000
hits_chunk: 25000
max_keys: 4000000
inter_thresh: 1.0
threshold_adc: 2.0
inst_threshold: 25.0
max_buckets: 1000
```

See `profiler/` for individual scripts to tune each parameter separately.

### CLI Flags

| Flag | Default | Description |
|---|---|---|
| `--data` | `mpvmpr_20.h5` | Input HDF5 file with particle step data |
| `--config` | `config/cubic_wireplane_config.yaml` | Detector configuration YAML |
| `--production-config` | none | Load optimized params from profiler config YAML |
| `--dataset` | `sim` | Dataset name prefix for output files |
| `--outdir` | `.` | Output directory (creates `sensor/`, `seg/`, `inst/` subdirs) |
| `--events` | all | Number of events to process |
| `--events-per-file` | 1000 | Events per output HDF5 file |
| `--threshold-adc` | 2.0 | Minimum signal amplitude to store (ADC) |
| `--workers` | 2 | Number of save worker threads (0 = serial) |
| `--noise` | off | Enable intrinsic noise |
| `--electronics` | off | Enable RC-RC electronics response |
| `--no-digitize` | on | Disable ADC digitization |
| `--no-track-hits` | on | Disable per-instance decomposition (inst file) |
| `--sce` | off | Path to SCE HDF5 map for E-field distortions |
| `--total-pad` | 500,000 | Max deposits per volume (sets JIT compiled shape) |
| `--response-chunk` | 50,000 | Deposits per response fori_loop batch (must divide total-pad) |
| `--hits-chunk` | 25,000 | Deposits per track-hits fori_loop batch (must divide total-pad) |
| `--max-keys` | 4,000,000 | Track-hits merge state capacity per plane (RuntimeError if exceeded) |
| `--inter-thresh` | 1.0 | Track-hits intermediate pruning threshold (electrons) |
| `--inst-threshold` | 25.0 | Charge threshold for inst (per-instance) entries (electrons) |
| `--max-buckets` | 1,000 | Max active buckets per plane (bucketed mode) |
| `--group-size` | 5 | Deposits per inst (correspondence) group |
| `--gap-threshold` | 5.0 | Group split threshold in mm |
| `--seed` | 42 | Random seed for noise generation |

## Pipeline

For each event, `run_batch.py` performs:

1. **Load** particle step data from HDF5 (positions, energy deposits, angles, track IDs)
2. **Group** deposits into runs of `group_size` consecutive steps per track, split on spatial gaps and the cathode boundary (groups never span east/west sides)
3. **Simulate** detector response via `DetectorSimulator` (GPU JIT-compiled):
   - Charge recombination (EMB or Modified Box model)
   - Electron drift with lifetime attenuation
   - Diffusion-convolved wire response (DCT-based kernel interpolation)
   - Q_s fractions computed inside JIT from recombined charges
   - Optional: electronics response, noise, ADC digitization
4. **Save** to three HDF5 file types (offloaded to worker threads):
   - `sensor/`: sparse thresholded raw readout
   - `seg/`: compact 3D truth deposits
   - `inst/`: per-instance sensor decomposition (group-level 3D-to-2D mapping)

The canonical layout in ``docs/DATASET_DESIGN.md`` (in
``particle-imaging-models/``) defines a fourth directory, ``labl/``,
holding per-track labels. It is produced **separately**, not by
``run_batch.py`` — see "Generating labl/" below.

## Threading Architecture

With `--workers N` (default 2), save work is offloaded to background threads:

```
Main thread:   load → GPU sim → queue   load → GPU sim → queue   ...
Worker 1:      CSR encode → write       CSR encode → write       ...
Worker 2:           CSR encode → write       CSR encode → write  ...
```

- **CSR encoding** (numpy) releases the GIL — multiple workers encode in parallel
- **HDF5 writes** serialize through a file lock — one write at a time
- **GPU simulation** releases the GIL during `block_until_ready()` — workers run concurrently
- Queue depth = workers + 2 to absorb event size variation

With 2 workers on typical events (~170K deposits): **~1.3s/event** (vs 2.9s serial).

## Viewing Output

The `view_production.ipynb` notebook loads and visualizes production output without running any simulation. It only needs the output HDF5 files — no YAML config or `generate_detector` required.

```python
from production.load import (
    get_file_paths, build_viz_config,
    load_event_sensor, load_event_seg, load_event_inst,
)

sensor_path, seg_path, inst_path = get_file_paths('output/', 'myrun', file_index=0)
viz_config = build_viz_config(sensor_path)  # minimal config from HDF5 metadata
dense_signals, attrs, pedestals = load_event_sensor(sensor_path, event_idx=0)
# pedestals is {(vol, plane): int} if digitized, None otherwise
# To get signed ADC: signal = dense_signals[(v,p)].astype(int) - pedestals[(v,p)]
seg = load_event_seg(seg_path, event_idx=0)         # list of per-volume dicts
track_hits, truth_dense, g2t, s2g, qs = load_event_inst(
    inst_path, event_idx=0, num_time_steps=2701)
# g2t, s2g, qs are lists of per-volume arrays (segment_to_group is the
# per-deposit group id, qs_fractions is the per-deposit charge weight).
```

---

## Output File Format

Three file types per batch, split by `events_per_file`:

```
{dataset}_sensor_{NNNN}.h5  — raw sensor readout (sparse wire/pixel signals)
{dataset}_seg_{NNNN}.h5     — 3D truth deposits (segment data)
{dataset}_inst_{NNNN}.h5    — per-instance sensor decomposition
```

### 1. Sensor File (`_sensor_`)

Sparse thresholded raw readout after full detector simulation.

```
/config/
    attrs: dataset_name, source_file, n_events, global_event_offset,
           num_time_steps, time_step_us, electrons_per_adc,
           velocity_cm_us, lifetime_us, recombination_model,
           include_noise, include_electronics, include_digitize,
           threshold_adc, n_bits (if digitized)
    num_wires           (2, 3) int32
    pedestals           (2, 3) int32    per-plane pedestal (if digitized)

/event_{NNN}/
    attrs: source_event_idx, n_volumes, n_vol0, n_vol1, ...
    volume_N/{plane}/                  per-volume groups; plane labels
                                       come from cfg.plane_names (e.g.
                                       U/V/Y for wire, "Pixel" for pixel)
        delta_wire      (P,) int16     delta-encoded wire indices
        delta_time      (P,) int16     delta-encoded time indices
        values          (P,) uint16    unsigned ADC (pedestal added) if digitized
                        (P,) float32   signal amplitude (ADC) if not digitized
        attrs: wire_start, time_start, n_pixels, pedestal (if digitized)
```

**Decode:**
```python
wires = wire_start + np.cumsum(delta_wire)
times = time_start + np.cumsum(delta_time)
# If digitized (values.dtype == uint16):
signal_adc = values.astype(np.int32) - pedestal  # signed ADC
```

### 2. Seg File (`_seg_`)

Pure 3D truth physics. Deposit-level scalars only — no instance
identifiers, no per-track metadata, no group machinery. Per the design
doc (in ``particle-imaging-models/docs/DATASET_DESIGN.md``), seg is
loadable standalone for SSL on 3D deposits or per-deposit physics
regression; any labeling / correspondence goes through ``labl/`` or
``inst/``.

```
/config/
    attrs: dataset_name, source_file, n_events, global_event_offset,
           group_size, gap_threshold_mm
    num_wires           (2, 3) int32

/event_{NNN}/
    attrs: source_event_idx
    volume_N/
        attrs: n_actual, pos_origin_x/y/z, pos_step_mm
        positions   (N, 3) uint16    voxelized at pos_step_mm
        de          (N,) float16     energy deposit in MeV
        dx          (N,) float16     step length in mm
        theta       (N,) float16     polar angle
        phi         (N,) float16     azimuthal angle
        t0_us       (N,) float16     step time (µs)
        charge      (N,) float32     recombined charge
        photons     (N,) float32     scintillation photons
```

**NOT stored here** (moved to inst/ or labl/ per design):
``track_ids``, ``group_ids``, ``group_to_track``, ``qs_fractions``,
``pdg``, ``interaction_ids``, ``ancestor_track_ids``,
``original_indices``.

**Decode positions:**
```python
positions_mm = positions.astype(np.float32) * pos_step_mm + \
               np.array([pos_origin_x, pos_origin_y, pos_origin_z])
```

### 3. Inst File (`_inst_`)

Per-instance sensor decomposition. Owns all group-related machinery:
each deposit's group assignment, the per-group → track lookup, the
within-group charge-fraction weights, and the CSR-encoded per-pixel
entries (which groups contributed to which pixels).

```
/config/
    attrs: dataset_name, source_file, n_events, global_event_offset,
           group_size, gap_threshold_mm, num_time_steps, threshold
    num_wires           (2, 3) int32

/event_{NNN}/
    attrs: source_event_idx, threshold
    volume_N/
        attrs: n_actual, n_groups
        segment_to_group  (N,) int32     per-deposit: which group each
                                         deposit belongs to. Row-aligned
                                         with seg deposits in volume N.
        qs_fractions      (N,) float16   per-deposit: each deposit's
                                         share of its group's recombined
                                         charge. Used for deposit-level
                                         disaggregation when traversing
                                         inst -> seg.
        group_to_track    (G,) int32     per-group: Geant4 track_id of
                                         each group.
        {plane_label}/                   one subgroup per readout plane
            group_ids       (G_p,) int32   active groups on this plane
            group_sizes     (G_p,) uint8   entries per group
            center_wires    (G_p,) int16   wire idx of peak-charge pixel
            center_times    (G_p,) int16   time idx of peak-charge pixel
            peak_charges    (G_p,) float32 charge at peak pixel (e-)
            delta_wires     (N_p,) int8    wire offset from group center
            delta_times     (N_p,) int8    time offset from group center
            charges_u16     (N_p,) uint16  charge as fraction of peak
                                           (x65535)
            attrs: n_groups_plane, n_entries
```

**Decode:**
```python
group_starts = np.cumsum(group_sizes) - group_sizes
# For group i, entries are [group_starts[i] : group_starts[i] + group_sizes[i]]
wire = center_wires[i] + delta_wires[j]
time = center_times[i] + delta_times[j]
charge = peak_charges[i] * charges_u16[j] / 65535.0
```

---

### 4. Labl File (`_labl_`)

Per-track labels and the per-deposit → track_id foreign key. **Not
produced by `run_batch.py`** — generated separately (currently by the
stop-gap script `make_labl.py`, see below). Lives under
`{outdir}/labl/{dataset}_labl_{NNNN}.h5`.

```
/config/
    attrs: dataset_name, source_file, n_events, n_volumes,
           label_names, source, generator, ...

/event_{NNN}/
    volume_N/
        # Per-deposit FK (N,) — row-aligned with seg deposits for vol N
        segment_to_track     (N,) int32    deposit i -> Geant4 track_id

        # Per-unique-track (T,) dimension table
        track_ids            (T,) int32    primary key: unique tracks
        track_pdg            (T,) int32    raw PDG code per track
        track_interaction    (T,) int32    raw interaction_id per track
        track_ancestor       (T,) int32    raw ancestor track_id per track
        track_cluster        (T,) int32    dummy (= track_ids) for now
```

**Design rationale.** This is a two-section layout:
- The per-deposit `segment_to_track` gives the track_id for each deposit
  directly (no subgroup; its shape N_deposits distinguishes it).
- The per-unique-track columns `track_{pdg, interaction, ancestor,
  cluster}` are a dimension table keyed by `track_ids`, avoiding
  per-track metadata being broadcast N times across deposits.

**Label lookup (per-deposit):**
```python
# For deposit i in volume v:
#   track_id = labl_v.segment_to_track[i]
#   j = np.searchsorted(sort(labl_v.track_ids), track_id)
#   pdg = labl_v.track_pdg[j]
```

**Generating labl — `production/make_labl.py`**

`make_labl.py` is a temporary stand-in for a proper edepsim-side labl
writer. It pulls the deposit → group assignment from `inst/`, the
group → track_id map from `inst/`, and the per-track metadata (pdg,
interaction, ancestor) from the original edepsim HDF5 source file.

```bash
python3 production/make_labl.py --outdir dataset_20 --source out.h5
```

- `--outdir`: dataset directory; must contain `inst/`. Creates `labl/`
  alongside.
- `--source`: edepsim HDF5 file. Defaults to the `source_file` attr
  recorded in the inst config.
- `--dataset`: filename prefix (default `sim`).

Runtime is trivial (<2 s per 20-event file). This script is explicitly
out of the simulation pipeline and **not JIT-compiled** — replace with
an edepsim-side integrated writer when productionizing.

## Bidirectional Correspondence

Correspondence between 3D deposits (seg) and 2D pixels (sensor) is
carried by **group ids only** — `group_to_track` is a *label*, not part
of the mapping. The three inst arrays below are all you need:

| Array | Shape | Meaning |
|---|---|---|
| `segment_to_group` (`s2g`) | `(N_dep,)` | per-deposit → group id (row-aligned with seg[v]) |
| `qs_fractions` (`qs`)      | `(N_dep,)` | per-deposit share of its group's recombined charge (sums to ~1 per group) |
| per-plane CSR (`group_ids`, `delta_wires`/`delta_times`, `peak_charges`, `charges_u16`) | `(N_entries,)` flat | each entry = one group's charge contribution at one pixel |

`load_correspondence(inst_path, event_idx, v)` returns a decoded
per-volume dict:

```python
from production.load import load_correspondence, segment_charge_per_plane

corr = load_correspondence(inst_p, event_idx=0, v=0)
# corr['s2g'], corr['qs'], corr['g2t'],
# corr['planes'] = {'U': {'wire','time','charge','gid'}, 'V': ..., 'Y': ...}
#                  (pixel readouts replace 'wire' with 'pixel_y'+'pixel_z')
```

**Forward (deposit → pixels):**
```python
i = deposit_idx
plane = corr['planes']['U']                          # or V/Y/Pixel
g = corr['s2g'][i]
mask = plane['gid'] == g
wires  = plane['wire'][mask]
times  = plane['time'][mask]
dep_ch = corr['qs'][i] * plane['charge'][mask]       # this deposit's share (e-)
```
*Empty is legitimate.* Groups whose peak charge fell below
`inst_threshold` (default 25 e⁻) have no plane entries, so deposits in
those groups produce no pixels (~70% of deposits in typical events).

**Backward (pixel → deposits):**
```python
plane = corr['planes']['U']
hit = (plane['wire'] == w_q) & (plane['time'] == t_q)
gids_at_pixel = np.unique(plane['gid'][hit])         # usually >1 group
deposits = np.concatenate([np.where(corr['s2g'] == g)[0]
                           for g in gids_at_pixel])
```
A single pixel is typically shared by several groups (≥3 groups at more
than half of active pixels in dense events). Per-group charge
contributions at this pixel are `plane['charge'][hit]` (one row per
contributing group).

**Per-segment total charge landing on a plane** (fast, vectorized):
```python
totals_U = segment_charge_per_plane(corr, 'U')   # (N_dep,) float32
# totals_U[i] = qs[i] * sum of group[s2g[i]]'s charge on plane U.
# Zero for deposits in sub-threshold groups.
```
For a typical event: `totals_U.sum()` is ~25–30% of `seg['charge'].sum()`
(the rest is lifetime attenuation + sub-threshold charge not written).

**Indexing note.** Group ids are **1-based** in `s2g` (`min=1`); entry
`group_to_track[0]` is an unused slot.

**Optional — track label for a group** (not correspondence):
```python
track_id = corr['g2t'][group_id]
```
For per-deposit track labels, prefer `labl/`'s `segment_to_track`
(direct deposit → track_id, no group indirection).

**Deriving labeled hits from correspondence:**
```python
from tools.track_hits import label_from_groups
result = label_from_groups(pk, gid, ch, count, group_to_track, max_time)
# result['labeled_hits'] (P, 3), result['labeled_track_ids'] (P,)
```

---

## Size Reference

Typical event with ~170K deposits, group_size=5, threshold=2.0 ADC:

| File | Per event | Per 1000 events |
|---|---|---|
| Response | ~2.4 MB | ~2.4 GB |
| Segments | ~1.3 MB | ~1.3 GB |
| Correspondence | ~8.0 MB | ~8.0 GB |
| **Total** | **~11.7 MB** | **~11.7 GB** |

Without correspondence (`--no-track-hits`): ~3.7 MB/event, ~3.7 GB per 1000 events.

## Performance

| Mode | Time/event | Throughput |
|---|---|---|
| Serial (with corr) | ~2.9s | ~0.3 events/s |
| 2 workers (with corr) | ~1.3s | ~0.8 events/s |
| 2 workers (no corr) | ~0.5s | ~2.0 events/s |

## Overflow Protection

Both `total_pad` and `max_keys` raise `RuntimeError` if exceeded:

- **total_pad overflow**: A volume has more deposits than `total_pad`. Raised during data loading (before simulation). Fix: increase `--total-pad` or run `profiler.find_optimal_pad`.
- **max_keys overflow**: Track-hits merge state exceeds `max_keys` capacity for a plane. Raised after simulation. Fix: increase `--max-keys` or run `profiler.setup_production`.

Use `--production-config` with a profiler-generated config to avoid both.
