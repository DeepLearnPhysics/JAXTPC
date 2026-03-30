# JAXTPC Production Pipeline

Batch simulation of particle events in a liquid argon TPC, producing structured HDF5 output for downstream analysis and ML training.

## Contents

```
production/
├── run_batch.py              # Main batch simulation script
├── save.py                   # HDF5 save functions (resp/seg/corr encoding)
├── load.py                   # HDF5 load/decode functions
├── view_production.ipynb     # Visualize production output (no simulation needed)
└── README.md                 # This file
```

## Usage

From the project root:

```bash
# Basic run (10 events, 2 save workers, digitization on, noise/electronics off)
python3 production/run_batch.py --data events.h5 --events 10

# Full options
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

### CLI Flags

| Flag | Default | Description |
|---|---|---|
| `--data` | `mpvmpr_20.h5` | Input HDF5 file with particle step data |
| `--config` | `config/cubic_wireplane_config.yaml` | Detector configuration YAML |
| `--dataset` | `sim` | Dataset name prefix for output files |
| `--outdir` | `.` | Output directory (creates `resp/`, `seg/`, `corr/` subdirs) |
| `--events` | all | Number of events to process |
| `--events-per-file` | 1000 | Events per output HDF5 file |
| `--threshold-adc` | 2.0 | Minimum signal amplitude to store (ADC) |
| `--workers` | 2 | Number of save worker threads (0 = serial) |
| `--noise` | off | Enable intrinsic noise |
| `--electronics` | off | Enable RC-RC electronics response |
| `--no-digitize` | on | Disable ADC digitization |
| `--no-track-hits` | on | Disable track correspondence |
| `--sce` | off | Path to SCE HDF5 map for E-field distortions |
| `--group-size` | 5 | Deposits per correspondence group |
| `--gap-threshold` | 5.0 | Group split threshold in mm |
| `--corr-threshold` | 25.0 | Charge threshold for correspondence entries (electrons) |
| `--total-pad` | 500,000 | Max deposits per side (sets JIT compiled shape) |
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
   - Response: sparse thresholded wire signals
   - Segments: compact 3D truth deposits
   - Correspondence: group-level 3D-to-2D mapping

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
from production.load import get_file_paths, build_viz_config, load_event_resp, load_event_seg, load_event_corr

resp_path, seg_path, corr_path = get_file_paths('output/', 'myrun', file_index=0)
viz_config = build_viz_config(resp_path)  # minimal config from HDF5 metadata
dense_signals, attrs = load_event_resp(resp_path, event_idx=0)
seg = load_event_seg(seg_path, event_idx=0)
track_hits, truth_dense, g2t = load_event_corr(corr_path, event_idx=0, num_time_steps=2701)
```

---

## Output File Format

Three file types per batch, split by `events_per_file`:

```
{dataset}_resp_{NNNN}.h5   — detector response (sparse wire signals)
{dataset}_seg_{NNNN}.h5    — 3D truth deposits (segment data)
{dataset}_corr_{NNNN}.h5   — 3D-to-2D correspondence map
```

### 1. Response File (`_resp_`)

Sparse thresholded wire signals after full detector simulation.

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

**Decode:**
```python
wires = wire_start + np.cumsum(delta_wire)
times = time_start + np.cumsum(delta_time)
```

### 2. Segment File (`_seg_`)

3D truth deposits. Array index = segment ID, referenced by correspondence.

```
/config/
    attrs: dataset_name, source_file, n_events, global_event_offset,
           group_size, gap_threshold_mm
    num_wires           (2, 3) int32

/event_{NNN}/
    attrs: source_event_idx, n_deposits, n_east, n_west, n_groups,
           pos_origin_x/y/z, pos_step_mm
    positions           (N, 3) uint16    voxelized at pos_step_mm resolution
    de                  (N,) float16     energy deposit in MeV
    dx                  (N,) float16     step length in mm
    theta               (N,) float16     polar angle
    phi                 (N,) float16     azimuthal angle
    track_ids           (N,) int32       Geant4 particle track ID
    group_ids           (N,) int32       group assignment
    group_to_track      (G,) int32       group ID -> track ID lookup
    qs_fractions        (N,) float16     deposit charge fraction within group
```

**Decode positions:**
```python
positions_mm = positions.astype(np.float32) * pos_step_mm + np.array([pos_origin_x, pos_origin_y, pos_origin_z])
```

**Group assignment:** Consecutive runs of `group_size` deposits per track, split on spatial gaps > `gap_threshold_mm` and on the cathode boundary (x=0). `qs_fractions[i]` = deposit's share of its group's recombined charge, used for disaggregation.

### 3. Correspondence File (`_corr_`)

3D-to-2D correspondence: which groups contributed to which pixels, stored in CSR format.

```
/config/
    attrs: dataset_name, source_file, n_events, global_event_offset,
           group_size, gap_threshold_mm, num_time_steps, threshold
    num_wires           (2, 3) int32

/event_{NNN}/
    attrs: source_event_idx, n_deposits, n_groups, threshold
    group_to_track      (G,) int32

    {plane}/                              6 planes: east_U/V/Y, west_U/V/Y
        group_ids       (G_p,) int32      active groups on this plane
        group_sizes     (G_p,) uint8      entries per group
        center_wires    (G_p,) int16      wire index of peak-charge pixel
        center_times    (G_p,) int16      time index of peak-charge pixel
        peak_charges    (G_p,) float32    charge at peak pixel (electrons)
        delta_wires     (N_p,) int8       wire offset from group center
        delta_times     (N_p,) int8       time offset from group center
        charges_u16     (N_p,) uint16     charge as fraction of peak (x65535)
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

## Bidirectional Correspondence

**Forward (segment -> hits):**
```python
group = group_ids[deposit_idx]
# Find group in correspondence, decode entries -> (wire, time, charge)
deposit_charge_at_pixel = qs_fractions[deposit_idx] * group_charge_at_pixel
```

**Backward (hit -> segments):**
```python
# Scan correspondence entries for matching (wire, time)
track_id = group_to_track[group_id]
members = np.where(group_ids == group_id)[0]  # constituent deposits
```

**Deriving track labels (not stored, computed from correspondence):**
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

## Internal Parameters

| Parameter | Default | Description |
|---|---|---|
| `inter_thresh` | 1.0 | Prune correspondence entries below this charge (electrons) during JIT merge |
| `max_keys` | 4,000,000 | Merge state capacity per plane. Warning printed if exceeded |
