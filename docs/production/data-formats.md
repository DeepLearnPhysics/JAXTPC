# Production data formats

The batch pipeline (`production/run_batch.py`) writes structured HDF5 that is
the authoritative interface for downstream analysis and ML training. This page
is the reference for the on-disk schema, the encodings, the compression codecs,
and the load recipes in `production/load.py`.

A batch produces **four file types**, split into files of `--events-per-file`
events each and written into per-type subdirectories:

```text
{outdir}/sensor/{dataset}_sensor_{NNNN}.h5
{outdir}/step/{dataset}_step_{NNNN}.h5
{outdir}/hits/{dataset}_hits_{NNNN}.h5
{outdir}/labl/{dataset}_labl_{NNNN}.h5     # produced separately, see below
```

| File | Prefix | Contents | Written by |
|---|---|---|---|
| **sensor** | `_sensor_` | Sparse thresholded raw readout after full detector sim; delta-encoded, `uint16` if digitized | `run_batch.py` |
| **step** | `_step_` | 3D truth deposits — positions + `de`/`dx`/`theta`/`phi`/`t0_us` + `charge`/`photons`; pure physics, no track or instance info | `run_batch.py` |
| **hits** | `_hits_` | Per-particle sensor decomposition + group machinery — `deposit_to_group`, `qs_fractions`, `group_to_track` per volume, per-plane CSR-encoded entries | `run_batch.py` (with track-hits on) |
| **labl** | `_labl_` | Per-track labels + per-deposit → `track_id` foreign key | `production/make_labl.py` (separate, temporary stand-in) |

Every event lives under an `event_{NNN}` group (`NNN` = index within the file,
zero-padded to 3 digits), and every event is split by volume into
`volume_{v}` subgroups. Wire planes are labelled `U`/`V`/`Y`; pixel volumes use
a single `Pixel` plane.

!!! info "Files each carry their own `/config` group"
    Metadata (num time steps, sampling, pedestals, `num_wires` per
    (volume, plane), volume ranges in mm, provenance) is written once per file
    under `/config`. `production/load.py:load_config` / `build_viz_config` read
    it, so viewing output needs **only the HDF5 file** — no YAML or
    `generate_detector`.

---

## Encodings

Two compact encodings recur across the formats.

**Delta encoding (sensor, hits).** Sparse coordinate arrays are sorted and
stored as successive differences (`np.diff(x, prepend=x[0])`), plus a scalar
`*_start`. Consecutive active elements differ by small integers, so the deltas
fit in `int16`/`int8` and compress well. Decode with a cumulative sum:

```python
wires = wire_start + np.cumsum(delta_wire)
times = time_start + np.cumsum(delta_time)
```

**CSR encoding (hits).** The per-plane group→pixel correspondence is stored in
compressed-sparse-row layout: one row per group, `group_sizes` giving the entry
count per group, and per-entry deltas from a per-group **center** (the
peak-charge pixel). Charges are stored as a fraction of the group's peak
(`uint16` ×65535 for wire, signed `int16` ×32767 for pixel), with the absolute
peak in `peak_charges`. This keeps per-entry payloads to 1–2 bytes.
`encode_correspondence_csr` (wire) and `encode_correspondence_csr_pixel`
(pixel) produce it; the `production/load.py` decoders invert it.

## Compression codecs

All datasets are compressed with the codec set by `production/save.py:set_codec`
(or `run_batch --codec`). The default is **`blosc-zstd`** (level 4 + byte
shuffle): smaller than gzip *and* faster on both read and write — gzip is
Pareto-dominated. Alternatives: `blosc-lz4hc` (gzip's size, ~4× faster reads),
`blosc-lz4` (fastest read+write, ~+19% size), `gzip`, `gzip-1`, `lzf`, `lz4`,
`zstd`.

!!! warning "Reading non-gzip output requires `import hdf5plugin`"
    The blosc/zstd/lz4 codecs are HDF5 filter plugins. `production/load.py`
    registers them automatically (it imports `hdf5plugin` at module load), but
    any ad-hoc `h5py` consumer must `import hdf5plugin` itself before opening a
    file, or reads of compressed datasets fail. A missing backend raises loudly
    on write — there is no silent downgrade to gzip.

---

## 1. Sensor (`_sensor_`)

Sparse thresholded raw readout after the full detector simulation (response →
optional electronics → noise → digitize). Only elements clearing
`threshold_adc` in absolute value are stored.

```text
/config/
    attrs: dataset_name, file_index, source_file, n_events,
           global_event_offset, num_time_steps, time_step_us,
           pre_window_us, post_window_us, electrons_per_adc,
           velocity_cm_us, lifetime_us, recombination_model,
           include_intrinsic_noise, include_coherent_noise,
           include_electronics, include_digitize, threshold_adc,
           n_volumes, readout_type, n_bits (if digitized)
    num_wires        (n_volumes, max_planes) int32
    volume_ranges    (n_volumes, 3, 2)       float32   mm, [vol][axis][min,max]
    pedestals        (n_volumes, max_planes) int32     if digitized

/event_{NNN}/
    attrs: source_event_idx, n_volumes, n_vol0, n_vol1, ...   (deposit counts)
    volume_{v}/{plane}/                     plane ∈ {U,V,Y} (wire) or Pixel
        # wire readout:
        delta_wire    (P,) int16
        delta_time    (P,) int16
        values        (P,) uint16    raw ADC (pedestal added) if digitized
                      (P,) float32   signal amplitude (ADC) otherwise
        attrs: wire_start, time_start, n_pixels, pedestal (if digitized)
        # pixel readout:
        delta_py, delta_pz, delta_time  (P,) int16
        values                          (P,) float32
        attrs: py_start, pz_start, time_start, n_pixels
```

!!! note "Units by readout"
    Digitized wire `values` are raw **ADC** (12-bit by default); non-digitized
    wire `values` are ADC-scale float. Pixel `values` are **ADC** directly (the
    chip gain is baked into the pixel kernel — pixel has no separate digitize
    step). See [units](../physics/units.md).

### Load recipe

```python
from production.load import load_event_sensor

# Non-digitized wire: dense arrays are safe.
signals, event_attrs, pedestals = load_event_sensor(sensor_path, event_idx=0)
# signals: {(vol, plane): (num_wires, num_time) float ndarray}, pedestals is None

# Digitized wire or any pixel: request sparse (dense is refused — see below).
signals, event_attrs, pedestals = load_event_sensor(
    sensor_path, event_idx=0, as_sparse=True)
# wire:  signals[(v,p)] = {'wire', 'time', 'values'}   values = raw ADC if digitized
# pixel: signals[(v,p)] = {'py', 'pz', 'time', 'values'}
# pedestals: {(vol, plane): int} if digitized, else None
```

!!! danger "Pedestal subtraction for signed ADC"
    Digitized `values` are stored as raw ADC = pedestal + signal, so a dense
    canvas would wash the zero background to `−pedestal`. `load_event_sensor`
    therefore **refuses** to densify digitized wire or any pixel data — pass
    `as_sparse=True` and subtract the pedestal yourself:
    ```python
    signal_adc = signals[(v, p)]['values'].astype(np.int32) - pedestals[(v, p)]
    ```

---

## 2. Step (`_step_`)

Pure 3D truth physics. Deposit-level scalars only — **no** instance identifiers,
per-track metadata, or group machinery (those live in `hits/` and `labl/`).
Positions are voxelised to `uint16` at `pos_step_mm`; physics scalars are
`float16`; sim-derived `charge`/`photons` are `float32`.

```text
/config/
    attrs: dataset_name, file_index, source_file, n_events,
           global_event_offset, group_size, gap_threshold_mm,
           n_volumes, readout_type
    num_wires        (n_volumes, max_planes) int32
    volume_ranges    (n_volumes, 3, 2)       float32   mm

/event_{NNN}/
    attrs: source_event_idx, n_volumes
    volume_{v}/
        attrs: n_actual, pos_origin_x/y/z, pos_step_mm
        positions   (N, 3) uint16    voxelised at pos_step_mm
        de          (N,) float16     energy deposit (MeV)
        dx          (N,) float16     step length (mm)
        theta       (N,) float16     polar angle
        phi         (N,) float16     azimuthal angle
        t0_us       (N,) float16     step time (µs)
        charge      (N,) float32     recombined charge (electrons)
        photons     (N,) float32     scintillation photons
```

Positions are transformed from the simulator's local frame back to **global**
coordinates before saving (see [coordinates](../concepts/coordinates.md)).
Decode: `positions_mm = positions * pos_step_mm + [pos_origin_x/y/z]`.

### Load recipe

```python
from production.load import load_event_step

volumes = load_event_step(step_path, event_idx=0)   # list of per-volume dicts
v0 = volumes[0]
# v0['positions_mm'] (N,3), v0['de'], v0['dx'], v0['theta'], v0['phi'],
# v0['t0_us'], v0['charge'], v0['photons'], v0['n_actual']
```

`load_event_step` reverses the voxelisation and returns `float32` arrays.

---

## 3. Hits (`_hits_`)

Per-particle sensor decomposition plus all group machinery: each deposit's group
assignment, the per-group → track lookup, the within-group charge-fraction
weights, and the CSR-encoded per-pixel entries. Written only when track-hits is
enabled (it is on by default; `--no-track-hits` skips this file). See
[track-hits](../truth/track-hits.md) for the correspondence semantics.

```text
/config/
    attrs: dataset_name, file_index, source_file, n_events,
           global_event_offset, group_size, gap_threshold_mm,
           num_time_steps, pre_window_us, post_window_us,
           n_volumes, readout_type
    num_wires        (n_volumes, max_planes) int32
    volume_ranges    (n_volumes, 3, 2)       float32

/event_{NNN}/
    attrs: source_event_idx, n_volumes, threshold
    volume_{v}/
        attrs: n_actual, n_groups
        deposit_to_group  (N,) int32     per-deposit group id; row-aligned
                                         with step deposits in volume v
        qs_fractions      (N,) float16   per-deposit share of its group's
                                         recombined charge (sums to ~1/group)
        group_to_track    (G,) int32     per-group Geant4 track_id (a label)
        {plane}/                         one subgroup per active readout plane
            group_ids       (G_p,) int32   active groups on this plane
            group_sizes     (G_p,) uint8   entries per group (CSR row lengths)
            center_wires    (G_p,) int16   peak-charge wire idx   (wire)
            center_py       (G_p,) int16   peak pixel-y idx       (pixel)
            center_pz       (G_p,) int16   peak pixel-z idx       (pixel)
            center_times    (G_p,) int16   peak time idx
            peak_charges    (G_p,) float32 charge at the peak pixel
            delta_wires     (N_p,) int8    wire offset from center (wire)
            delta_py        (N_p,) int8    pixel-y offset from center (pixel)
            delta_pz        (N_p,) int8    pixel-z offset from center (pixel)
            delta_times     (N_p,) int8    time offset from center
            charges_u16     (N_p,) uint16  charge / peak ×65535   (wire)
            charges_i16     (N_p,) int16   signed charge/peak ×32767 (pixel)
            attrs: n_groups_plane, n_entries
```

!!! warning "Group ids are 1-based"
    `deposit_to_group` (and the CSR `group_ids`) are **1-based**: `min == 1`,
    and `group_to_track[0]` is an unused slot. The per-group arrays are indexed
    with this in mind — `group_to_track[g]` is the track for group `g`.

Wire uses `charges_u16` (unsigned — diffusion-only truth); pixel uses
`charges_i16` (signed — post-response truth with bipolar induction, so a group's
charge at a pixel can be negative). The peak is the entry with the largest
**absolute** charge.

**Decode one CSR entry** (`i` = group index, `j` = flat entry index within it):

```python
group_starts = np.cumsum(group_sizes) - group_sizes
# wire:
wire   = center_wires[i] + delta_wires[j]
time   = center_times[i] + delta_times[j]
charge = peak_charges[i] * charges_u16[j] / 65535.0
# pixel:
py     = center_py[i] + delta_py[j]
pz     = center_pz[i] + delta_pz[j]
charge = peak_charges[i] * charges_i16[j] / 32767.0     # may be negative
```

### Load recipes

`load_event_hits` sums per-pixel charge (erasing group identity) and derives
labelled hits — convenient for a dense truth image:

```python
from production.load import load_event_hits

track_hits, truth_dense, g2t, d2g, qs = load_event_hits(
    hits_path, event_idx=0)   # num_time_steps/n_volumes read from /config if omitted
# track_hits[(v,p)] : labelled-hit result (from tools.track_hits.label_from_groups)
# truth_dense[(v,p)]: (num_wires, num_time) dense array (wire) or sparse dict (pixel)
# g2t, d2g, qs      : lists of per-volume arrays
#   d2g = deposit_to_group (per-deposit group id), qs = qs_fractions
```

### Correspondence API

For deposit↔pixel correspondence that **preserves group identity**, use
`load_correspondence` and `deposit_charge_per_plane` — the documented
correspondence interface:

```python
from production.load import load_correspondence, deposit_charge_per_plane

corr = load_correspondence(hits_path, event_idx=0, v=0)
# corr['d2g'] (N,), corr['qs'] (N,), corr['g2t'] (G,),
# corr['planes'] = {'U': {'wire','time','charge','gid'}, ...}
#                  (pixel readouts give 'pixel_y' + 'pixel_z' instead of 'wire')

# Forward — one deposit's pixels on plane U:
i = 0
plane = corr['planes']['U']
mask   = plane['gid'] == corr['d2g'][i]
wires  = plane['wire'][mask]
dep_ch = corr['qs'][i] * plane['charge'][mask]     # this deposit's share (e-)

# Per-deposit total charge landing on a plane (fast, vectorised):
totals_U = deposit_charge_per_plane(corr, 'U')     # (N,) float32
```

!!! note "Empty is legitimate"
    Groups whose peak charge fell below `corr_threshold` (the hits CSR cutoff)
    have no plane entries, so deposits in those groups produce no pixels. In
    typical events a large fraction of deposits are sub-threshold and correctly
    map to zero pixels.

---

## 4. Labl (`_labl_`)

Per-track labels and the per-deposit → `track_id` foreign key. **Not produced by
`run_batch.py`** — generated separately by `production/make_labl.py`, which is a
**temporary stand-in** for a proper edepsim-side writer (it reads `hits/` for
the deposit→group→track chain and the original edepsim source for per-track
metadata; it is not JIT-compiled and is out of the simulation pipeline).

```text
/config/
    attrs: dataset_name, source_file, n_events, n_volumes,
           label_names, source, generator, ...

/event_{NNN}/
    volume_{v}/
        deposit_to_track  (N,) int32   per-deposit → Geant4 track_id
                                        (row-aligned with step deposits in v)
        track_ids         (T,) int32   primary key: unique tracks
        track_pdg         (T,) int32   PDG code per track
        track_interaction (T,) int32   interaction_id per track
        track_ancestor    (T,) int32   ancestor track_id per track
        track_cluster     (T,) int32   dummy (= track_ids) for now
```

The per-deposit `deposit_to_track` gives a track directly (no group
indirection); the per-unique-track columns are a dimension table keyed by
`track_ids`. Generate with:

```bash
python3 production/make_labl.py --outdir {outdir} --source edepsim.h5
```

For per-deposit track labels, prefer `labl/`'s `deposit_to_track` over the
group indirection in `hits/`.

---

## See also

- [Capacities & overflow](../concepts/capacities.md) — the array budgets these
  files' sizes depend on, and how the sim fails when they are exceeded.
- [Units](../physics/units.md) — ENC (wire) vs ADC (pixel), and which threshold
  is in which unit.
- [Track-hits](../truth/track-hits.md) — the group→track correspondence written
  into the hits file.
- [Profiler](profiler.md) — sizing `total_pad`/`max_keys`/`maxg` and the
  `threshold_adc`/`corr_threshold` cutoffs used above.
