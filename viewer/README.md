# JAXTPC Viewer

Interactive 3D/2D visualization and GIF export for JAXTPC production data.

## Files

| File | Description |
|------|-------------|
| `serve_viewer.py` | Local HTTP server with byte-range support for HDF5 files |
| `index.html` | Viewer HTML entry point |
| `viewer.js` | Main application (Three.js 3D + Canvas2D panels) |
| `shaders.js` | GLSL vertex/fragment shaders (ES module) |
| `colormaps.js` | Colormap definitions and interpolation (ES module) |
| `viewer.css` | Styles with dark and light mode support |
| `h5_worker.js` | Web Worker for streaming HDF5 reads via h5wasm |
| `export_gif.py` | Standalone rotating 3D GIF/MP4 generator |

## Interactive Viewer

Serves production HDF5 files (seg/corr/resp) and opens a browser-based viewer
with 3D segment display, 2D wire response panels, correspondence highlighting,
drift animation, and track/PDG/ancestor/interaction color modes.

### Usage

```bash
# Auto-detect dataset, open browser
python3 viewer/serve_viewer.py production_run/ --open

# Specify dataset and port
python3 viewer/serve_viewer.py production_run/ --dataset myrun --port 9000
```

Supports both flat directories (`production_run/*.h5`) and subdirectory layouts
(`production_run/seg/`, `production_run/corr/`, `production_run/resp/`).

### Controls

- **View modes**: HITS (truth segments), RESP (wire response), OPTICAL (light)
- **Color modes**: dE (energy deposit) or categorical (Track, PDG, Ancestor, Interaction)
- **Correspondence**: hover 3D segments to highlight 2D pixels (and vice versa)
- **Drift animation**: play/pause animated charge drift toward anodes
- **Volume selection**: view all volumes or a single volume
- **Track filter**: isolate specific tracks by ID or category
- **Double-click**: expand a 2D panel to full size
- **Save/Copy**: download PNG or copy to clipboard (buttons on each panel)
- **Theme**: dark/light mode toggle
- **Settings**: dE emphasis, drift speed, response gamma, optical thresholds

## GIF Export

Generates a rotating 3D point cloud GIF cycling through color modes:
Energy Deposit, Track ID, PDG, Ancestor ID, Interaction ID.

### Usage

```bash
# Default: 12s rotation, 30fps, 1440x1440, 100k points
python3 viewer/export_gif.py path/to/sim_seg_0000.h5 --event 0

# Custom settings
python3 viewer/export_gif.py path/to/sim_seg_0000.h5 \
    --event 0 \
    --duration 6 \
    --fps 30 \
    --size 1080 1080 \
    --dpi 200 \
    --max-points 80000 \
    --output my_event.gif

# Single volume, light background
python3 viewer/export_gif.py path/to/sim_seg_0000.h5 -e 0 -v 0 --light

# MP4 output (requires ffmpeg)
python3 viewer/export_gif.py path/to/sim_seg_0000.h5 -e 0 -o event.mp4
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--event, -e` | 0 | Event index |
| `--volume, -v` | all | Volume index (omit for all) |
| `--output, -o` | `jaxtpc_3d.gif` | Output file (.gif, .mp4, .webm) |
| `--duration` | 12.0 | Seconds per full rotation |
| `--rotations` | 1 | Number of 360 rotations |
| `--fps` | 30 | Frames per second |
| `--max-points` | 100000 | Max segments to render |
| `--dpi` | 200 | Render resolution (affects point size) |
| `--size` | 1440 1440 | Output pixel dimensions |
| `--emph-pow` | 5.0 | dE emphasis power (steepness) |
| `--emph-amt` | 0.75 | dE emphasis amount (0=uniform, 1=full) |
| `--light` | off | Light background mode |

### Performance

Render time scales with `--max-points` (not resolution). At 100k points:
~1.8s/frame, so a 12s/30fps GIF takes ~11 minutes.

## Dependencies

**Interactive viewer** (browser): no installation needed (loads h5wasm from CDN).

**GIF export** (Python): `numpy`, `h5py`, `matplotlib`, `Pillow` (all standard scientific Python).
