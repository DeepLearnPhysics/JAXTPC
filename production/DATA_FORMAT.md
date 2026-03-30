# JAXTPC Production Output Format

The production batch pipeline (`run_batch.py`) simulates particle interactions in a liquid argon TPC and produces three output files per batch:

```
{dataset}_resp_{NNNN}.h5   — detector response (sparse wire signals)
{dataset}_seg_{NNNN}.h5    — 3D truth deposits (segment data)
{dataset}_corr_{NNNN}.h5   — 3D-to-2D correspondence map
```

Each file contains multiple events indexed as `event_000`, `event_001`, etc. Files are split by `events_per_file` (default 1000).

## Pipeline Overview

For each event, `run_batch.py` performs:

1. **Load** particle step data from HDF5 (positions, energy deposits, angles, track IDs)
2. **Group** deposits into runs of `group_size` consecutive steps per track, split on spatial gaps
3. **Simulate** detector response via `DetectorSimulator`:
   - Charge recombination (EMB or Modified Box model)
   - Electron drift with lifetime attenuation
   - Diffusion-convolved wire response (DCT-based kernel interpolation)
   - Optional: electronics response (RC-RC convolution), noise, digitization
4. **Save** three output files:
   - Response: sparse thresholded wire signals (delta-encoded + lzf)
   - Segments: compact 3D truth deposits (uint16 positions + float16 physics)
   - Correspondence: group-level 3D-to-2D mapping (CSR + delta + uint16/peak encoding)

---

## 1. Response File (`_resp_`)

Sparse thresholded wire signals after full detector simulation.

### Schema

```
/config/
    attrs: dataset_name, source_file, n_events, global_event_offset,
           num_time_steps, time_step_us, electrons_per_adc,
           velocity_cm_us, lifetime_us, recombination_model,
           include_noise, include_electronics, include_digitize,
           threshold_adc
    num_wires           (2, 3) int32

/event_{NNN}/
    attrs: source_event_idx, n_deposits, n_east, n_west
    {plane}/                           6 planes: east_U/V/Y, west_U/V/Y
        delta_wire      (P,) int16     delta-encoded wire indices
        delta_time      (P,) int16     delta-encoded time indices
        values          (P,) float32   signal amplitude (ADC)
        attrs: wire_start, time_start, n_pixels
```

### Decode

```python
wires = wire_start + np.cumsum(delta_wire)   # wire_start from attrs, deltas from 0
times = time_start + np.cumsum(delta_time)   # time_start from attrs
# wires[i], times[i], values[i] = one pixel's signal
```

### Units

- `values`: ADC counts (after electronics response and digitization)
- `threshold_adc`: minimum |signal| to keep (in ADC)

---

## 2. Segment File (`_seg_`)

The 3D truth deposits from the particle simulation (Geant4 steps). Array index = segment ID, referenced by the correspondence file.

### Schema

```
/config/
    attrs: dataset_name, source_file, n_events, global_event_offset,
           group_size, gap_threshold_mm
    num_wires           (2, 3) int32

/event_{NNN}/
    attrs: source_event_idx, n_deposits, n_east, n_west, n_groups
    positions           (N, 3) uint16    voxelized at pos_step_mm resolution
    de                  (N,) float16     energy deposit in MeV
    dx                  (N,) float16     step length in mm
    theta               (N,) float16     polar angle of step direction
    phi                 (N,) float16     azimuthal angle of step direction
    track_ids           (N,) int32       Geant4 particle track ID
    group_ids           (N,) int32       group assignment (see below)
    group_to_track      (G,) int32       group ID -> track ID lookup
    qs_fractions        (N,) float16     deposit charge fraction within group
    attrs: pos_origin_x/y/z, pos_step_mm (for decoding positions)
```

### Position Decoding

```python
positions_mm = positions.astype(np.float32) * pos_step_mm + np.array([pos_origin_x, pos_origin_y, pos_origin_z])
```

### Group Assignment

Deposits are grouped into consecutive runs of `group_size` (default 5) deposits along each particle track. Groups are split on spatial gaps exceeding `gap_threshold_mm` (default 5mm) to handle neutral particles that jump between distant interaction points.

- `group_ids[i]` = which group deposit `i` belongs to
- `group_to_track[g]` = the track ID for group `g`
- `qs_fractions[i]` = `recomb(dE_i, dx_i) / sum_group(recomb(dE, dx))` -- this deposit's share of its group's charge. Used for disaggregation. Plane-independent since attenuation is constant within a group.

---

## 3. Correspondence File (`_corr_`)

The 3D-to-2D correspondence map: for each wire plane, which groups of deposits contributed to which pixels, and how much charge.

Stored in CSR (compressed sparse row) format with delta-encoded pixel positions and uint16-quantized charges.

### Schema

```
/config/
    attrs: dataset_name, source_file, n_events, global_event_offset,
           group_size, gap_threshold_mm, num_time_steps, threshold
    num_wires           (2, 3) int32

/event_{NNN}/
    attrs: source_event_idx, n_deposits, n_groups, threshold
    group_to_track      (G,) int32        group -> track lookup

    {plane}/                              6 planes: east_U/V/Y, west_U/V/Y
        --- per-group arrays (G_p groups on this plane) ---
        group_ids       (G_p,) int32      which groups are active
        group_sizes     (G_p,) uint8      entries per group (max 255)
        center_wires    (G_p,) int16      wire index of peak-charge pixel
        center_times    (G_p,) int16      time index of peak-charge pixel
        peak_charges    (G_p,) float32    charge at peak pixel (electrons)

        --- per-entry arrays (N_p total entries) ---
        delta_wires     (N_p,) int8       wire offset from group center
        delta_times     (N_p,) int8       time offset from group center
        charges_u16     (N_p,) uint16     charge as fraction of peak (x65535)

        attrs: n_groups_plane (= G_p), n_entries (= N_p)
```

All datasets are gzip-compressed.

### Decode

```python
# Reconstruct pixel position for entry j in group i:
wire = center_wires[i] + delta_wires[j]
time = center_times[i] + delta_times[j]

# Reconstruct charge (in electrons):
charge = peak_charges[i] * charges_u16[j] / 65535.0

# Entry-to-group mapping uses cumulative sum of group_sizes:
group_starts = np.cumsum(group_sizes) - group_sizes  # CSR row pointers
# Entries [group_starts[i] : group_starts[i] + group_sizes[i]] belong to group i
```

### Units

- All charges are in **electrons** (not ADC). The correspondence operates on recombined charge before the response kernel.
- `inter_thresh` (default 1.0 electron) is applied during the merge inside the simulation.

---

## Bidirectional Correspondence

### Forward: Segment -> Hits

*Given a 3D deposit, which 2D pixels does it affect?*

```python
# 1. Deposit -> group
group = group_ids[deposit_idx]

# 2. Group -> pixels on a plane
plane_data = corr['east_Y']
mask = plane_data['group_ids'][:] == group
# Decode the entries for matching groups -> list of (wire, time, charge)

# 3. Deposit's share at each pixel
fraction = qs_fractions[deposit_idx]
deposit_charge = fraction * group_charge_at_pixel
```

### Backward: Hit -> Segments

*Given a 2D pixel, which 3D deposits produced its signal?*

```python
# 1. Pixel -> groups (scan entries for matching wire/time)
pixel_wire, pixel_time = 985, 1986

# 2. For each matching group -> track
track_id = group_to_track[group_id]

# 3. For each group -> constituent deposits
members = np.where(group_ids == group_id)[0]
positions = positions_mm[members]  # 3D locations
```

### Deriving Track Hits

*Dominant track per pixel -- not stored, derived from correspondence:*

```python
from tools.track_hits import label_from_groups

result = label_from_groups(pk, gid, ch, count, group_to_track, max_time)
# result['labeled_hits']       -- (P, 3) float32 [wire, time, charge]
# result['labeled_track_ids']  -- (P,) int32
```

---

## Size Reference

Typical event with ~170k deposits, group_size=5:

| File | Size (gzipped) | Contents |
|---|---|---|
| Response | ~5-15 MB | Depends on noise, threshold |
| Segments | ~3.5 MB | 3D positions, dE, dx, angles, groups, qs_fractions |
| Correspondence (threshold=0) | ~9 MB | Full correspondence map |
| Correspondence (threshold=50) | ~6 MB | Pruned kernel tails, 0.4% charge lost |
| Correspondence (threshold=100) | ~5.5 MB | 0.8% charge lost |

For 1000 events: segments ~3.5 GB, correspondence ~6-9 GB depending on threshold.

---

## Parameters

| Parameter | Default | CLI Flag | Effect |
|---|---|---|---|
| `group_size` | 5 | `--group-size` | Deposits per group. Larger = more compression, coarser 3D resolution |
| `gap_threshold_mm` | 5.0 | `--gap-threshold` | Split groups on spatial jumps (neutrons/gammas) |
| `threshold_adc` | 5.0 | `--threshold-adc` | Minimum signal amplitude to store (ADC) |
| `corr_threshold` | 0.0 | `--corr-threshold` | Prune correspondence entries below this charge (electrons) |
| `total_pad` | 500,000 | `--total-pad` | Max deposits per side. Truncates if exceeded |
| `events_per_file` | 1000 | `--events-per-file` | Events per output HDF5 file |
| `inter_thresh` | 1.0 | (internal) | Prune entries below this charge during JIT merge |
| `max_keys` | 4,000,000 | (internal) | Merge state capacity per plane. Warning printed if exceeded |
