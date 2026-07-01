# Plotting helpers

JAXTPC ships two Matplotlib plotting modules for inspecting simulation output:

- `tools/visualization.py` — **wire** readout (per-plane wire-vs-time images and
  scatter plots, truth diffused charge, track labels, waveforms).
- `tools/pixel_visualization.py` — **pixel** readout (orthogonal projections,
  anode heatmaps, per-pixel waveforms, 3D voxel scatter).

Every function returns a Matplotlib `Figure`, so you can further style, save, or
embed it. All of them take the `SimConfig` (`simulator.config`) for detector
geometry — number of wires/pixels per plane, time-step, sampling window.

!!! note "Units on these plots"
    Wire signal plots label the colorbar **Signal (ADC)** and take thresholds in
    **ENC (electrons)**; pixel plots are in **ADC** throughout. The wire vs pixel
    unit split is by design — see [units](../physics/units.md) for the full
    threshold-units table (`inter_thresh` / `corr_threshold` / `threshold_adc`).

## The sparse / dense input contract

Both modules accept either **dense** arrays or **sparse** dicts, keyed by a
`(volume, plane)` tuple. The two readouts use different sparse layouts:

| Readout | Dense value | Sparse dict keys |
|---|---|---|
| **Wire** | `(num_wires, num_time)` array | `wire`, `time`, `values` |
| **Pixel** | `(num_py, num_pz, num_time)` array | `pixel_y`, `pixel_z`, `time`, `values` |

Produce the sparse form with `simulator.to_sparse(...)` (or
`tools.output.to_sparse`) and the dense form with `simulator.to_dense(...)`.

The two modules **signal the format differently**, so watch this:

- **Wire functions take an explicit `sparse=` flag.** Pass `sparse=True` when you
  hand them a sparse dict, `sparse=False` (the default) for dense arrays.
  Mismatching the flag and the data raises a `ValueError` from `_get_values`
  (e.g. `sparse=False but data is a dict`). Sparse input is drawn with a
  `scatter`; dense input with `imshow`.
- **Pixel functions auto-detect** dense vs sparse (`isinstance(data, dict)`), so
  there is no `sparse=` flag — pass whichever you have. Dense pixel arrays are
  thresholded into a sparse set internally before projecting.

## Wire plotters

### `visualize_wire_signals`

The main wire viewer: one panel per `(volume, plane)` on a grid, each showing
wire index (x) vs time in µs (y). Uses the diverging **`obsidian`** colormap by
default with a [`DeadbandNorm`](#deadbandnorm) so bipolar induction signals read
symmetrically and the near-zero noise floor is suppressed.

Key parameters:

| Parameter | Default | Meaning |
|---|---|---|
| `threshold_enc` | `0` | Deadband half-width in **ENC**; converted to ADC internally (`threshold_enc / electrons_per_adc`) |
| `gamma` | `0.2` | `DeadbandNorm` power-law exponent (`<1` expands weak signals) |
| `cmap` | `'obsidian'` | Colormap; `'obsidian'` resolves to the built-in diverging map |
| `sparse` | `False` | Set `True` for sparse-dict input |
| `point_size` | `None` | Scatter marker size (sparse only); auto-sized to ~1 data cell if `None` |

Color scaling is per **plane type**: U/V induction planes share an
auto-computed `[vmin, vmax]`, and the Y (collection) plane is forced symmetric
about zero. Empty/absent planes render a "(No data)" placeholder.

### `visualize_diffused_charge`

Views the **truth diffused charge** (ionization electrons projected to the wire
plane, pre-electronics) rather than the readout signal. Panels are masked below
`threshold` (default `100`), colored with `YlOrRd` on a dark background, and
scaled to the global 1st–99th percentile of above-threshold values. Set
`log_norm=True` for a log color scale. Same `(volume, plane)` grid and
`sparse=` contract as `visualize_wire_signals`.

### `visualize_track_labels` and `get_top_tracks_by_charge`

`get_top_tracks_by_charge(track_hits, top_n=20)` takes the finalized track-hits
dict from `simulator.finalize_track_hits(...)` and returns a list of
`(track_id, total_charge)` tuples, highest charge first (charge summed across all
planes).

`visualize_track_labels(track_hits, config, top_tracks_by_charge, max_tracks=15)`
scatters each labeled hit colored by its track ID: the top `max_tracks` get
distinct fixed colors (with a legend), all others get a hashed HSV color. Feed it
the list from `get_top_tracks_by_charge` so the legend and coloring agree.

### Focused inspectors

For drilling into one plane or a few wires:

- `visualize_single_plane(...)` — one `(vol, plane)` panel, dense or sparse.
- `visualize_diffused_charge_single_plane(...)` — one plane, **dense only**.
- `visualize_track_labels_single_plane(...)` — one plane's track labels.
- `visualize_waveforms(wire_signals, config, wire_indices, ...)` — signal vs
  time line plots for a list of wire indices (dense or sparse).
- `visualize_wire_planes(detector_config, ...)` — wire geometry (angle/spacing)
  colored by wire index. Note this one takes the **raw** `detector_config` dict
  from `generate_detector`, not `SimConfig`, since it needs the plane angles.

### Bucketed-mode viewer

`visualize_active_buckets(response_signals, config)` renders the tile occupancy
of the wire **bucketed accumulation** mode — it expects the raw
5-tuple `(buckets, num_active, compact_to_key, B1, B2)` values straight out of
`process_event` (not decoded signals), and draws each active `B1×B2` tile as a
rectangle colored by its total energy. Use it to sanity-check bucket sizing when
running with `use_bucketed` / `--bucketed`; see
[capacities](../concepts/capacities.md) for what the bucketed mode is and how to
size it.

## Pixel plotters

All pixel functions are keyed by `(vol_idx, 0)` (pixel readout has a single
plane) and auto-detect dense vs sparse input.

| Function | What it shows |
|---|---|
| `visualize_pixel_projections` | Three orthogonal projections — Y-Z, Y-Time, Z-Time — each summed (`reduce='sum'`) or maxed (`reduce='max'`) along the collapsed axis |
| `visualize_pixel_anode` | 2D anode heatmap (Y-Z), optionally restricted to a `time_range=(t0, t1)` window in µs |
| `visualize_pixel_waveforms` | Signal vs time for a list of `(py, pz)` pixel coordinates |
| `visualize_pixel_3d` | 3D scatter of pixel voxels (Z, Y, time) colored by `|signal|`; downsamples past `max_points` |
| `visualize_all_pixel_volumes` | Runs `visualize_pixel_projections` for every volume whose `readout_type == 'pixel'`; returns a list of figures |

Common options: `cmap` (default `'inferno'`), `log_norm`, `threshold`, and
`reduce`. Physical extents (cm, µs) are derived from the volume's `pixel_shape`,
`pixel_pitch_cm`, and `pixel_origins_cm`.

There is also `visualize_pixel_buckets(...)`, the pixel analog of
`visualize_active_buckets`. It expects a pixel **6-tuple**
`(buckets, num_active, compact_to_key, B1, B2, B3)` and projects the 3D tile grid
onto the three planes. The pixel bucketed path is a reference implementation and
is not wired into the default pixel pipeline — see
[capacities](../concepts/capacities.md).

## DeadbandNorm

`DeadbandNorm` (in `tools/visualization.py`) is a `matplotlib.colors.Normalize`
subclass used by the wire plotters. It applies a symmetric **power-law
compression** around zero with an optional **dead zone**:

- `gamma < 1` expands weak signals (near the deadband) and compresses strong ones
  near `vmin`/`vmax` — the default `gamma=0.2` makes faint induction wiggles
  visible next to a saturated collection hit.
- `deadband` sets a `±deadband` window mapped to the flat middle of the colormap,
  so sub-threshold noise disappears into the neutral color. The wire plotters set
  it from `threshold_enc` (converted to ADC).
- `dead_frac` (default `0.08`) is the fraction of the colorbar occupied by the
  dead zone.

It implements a matching `inverse()`, so colorbars show real data values at
evenly-spaced norm ticks.

## Example

```python
import jax
from tools.simulation import DetectorSimulator
from tools.geometry import generate_detector
from tools.loader import load_event
from tools.visualization import (
    visualize_wire_signals,
    visualize_track_labels,
    get_top_tracks_by_charge,
)

detector_config = generate_detector('config/cubic_wireplane_config.yaml')
sim = DetectorSimulator(detector_config, include_track_hits=True, include_digitize=True)

deposits = load_event('data.h5', sim.config, event_idx=0)
response_signals, track_hits_raw, deposits = sim.process_event(
    deposits, key=jax.random.PRNGKey(0))

# Sparse readout, drawn as a scatter (note sparse=True)
sparse = sim.to_sparse(response_signals)
fig = visualize_wire_signals(sparse, sim.config, threshold_enc=200, sparse=True)
fig.savefig('wire_signals.png', dpi=150, bbox_inches='tight')

# Track labels, colored by the highest-charge tracks
track_hits = sim.finalize_track_hits(track_hits_raw)
top = get_top_tracks_by_charge(track_hits, top_n=20)
fig = visualize_track_labels(track_hits, sim.config, top, max_tracks=15)
fig.savefig('track_labels.png', dpi=150, bbox_inches='tight')
```

For pixel output, swap the imports for `tools.pixel_visualization` and call
`visualize_pixel_projections(pixel_signals, sim.config, vol_idx=0)` — no
`sparse=` flag needed.
