# Interactive viewer & GIF export

Two tools render **production output** (the `step` / `hits` / `sensor` HDF5
files, plus the optional `labl`) without re-running the simulation:

- **`jaxtpc-viewer`** (`viewer/serve_viewer.py`) — a local HTTP server that
  streams the HDF5 files to a browser-based 3D/2D viewer.
- **`jaxtpc-export-gif`** (`viewer/export_gif.py`) — a standalone renderer that
  writes a rotating 3D GIF or MP4 of one event.

Both are installed as console scripts by `pip install -e .`; you can equally run
`python3 viewer/serve_viewer.py …` / `python3 viewer/export_gif.py …`.

## Interactive viewer

`jaxtpc-viewer` serves a production run directory and opens a browser viewer with
a 3D truth-segment display, 2D wire/pixel sensor panels, correspondence
highlighting, drift animation, and — when a `labl` file is present —
track/PDG/ancestor/interaction color modes.

```bash
# Auto-detect the dataset, open the browser
jaxtpc-viewer production_run/ --open

# Pick a dataset by name and a port
jaxtpc-viewer production_run/ --dataset myrun --port 9000
```

The viewer reads HDF5 over **HTTP byte-range** requests (it never uploads whole
files), so large sensor files load incrementally as you pan and zoom.

### Layout & dataset detection

It handles both **flat** directories (`production_run/*.h5`) and the
**subdirectory** layout (`production_run/{step,hits,sensor}/`, plus optional
`labl/`). Datasets are discovered by parsing filenames of the form
`{dataset}_{kind}_{batch}.h5`. If a directory holds exactly one dataset it is
selected automatically; otherwise the server lists the candidates and asks you to
pass `--dataset`. (`_lzf`-suffixed files are skipped during discovery.)

For **wire** readout all three of `step`/`hits`/`sensor` must be present; for
**pixel** readout the `sensor` file is optional (the sensor image is derivable
from the hits file), detected from `config/num_wires == 0` in the hits file.

### Options

| Flag | Short | Default | Description |
|---|---|---|---|
| `data_dir` | — | *(required)* | Directory of production HDF5 files |
| `--dataset` | `-d` | auto | Dataset name (auto-detected when only one is present) |
| `--port` | `-p` | `8765` | Server port |
| `--open` | `-o` | off | Open the browser automatically |

The server binds to `127.0.0.1` (localhost only) and prints the URL
(`http://127.0.0.1:<port>/`); press `Ctrl+C` to stop it.

### Controls

- **View modes:** HITS (truth segments), SENSOR (raw wire/pixel readout),
  OPTICAL (light).
- **Color modes:** dE (energy deposit), or categorical — Track, PDG, Ancestor,
  Interaction (these require a `labl` file; hidden if it is missing).
- **Correspondence:** hover a 3D segment to highlight the matching 2D pixels, and
  vice versa.
- **Drift animation:** play/pause animated charge drift toward the anodes.
- **Volume selection:** all volumes or a single volume.
- **Track filter:** isolate tracks by ID or category.
- **Double-click** a 2D panel to expand it to full size.
- **Save / Copy:** download a panel as PNG or copy it to the clipboard.
- **Theme:** dark / light toggle.
- **Settings:** dE emphasis, drift speed, sensor gamma, optical thresholds.

!!! note "No install needed in the browser"
    The frontend loads `h5wasm` from a CDN, so the browser side has no local
    dependencies. If a `labl` file is absent the LABEL color mode and per-track
    filters are simply hidden — everything else still works.

## GIF export

`jaxtpc-export-gif` reads a **step** HDF5 file directly and renders a rotating 3D
point cloud with Matplotlib, cycling through color modes:
**Energy Deposit → Track ID → PDG → Ancestor ID → Interaction ID**, one segment
of the rotation per mode.

The categorical modes pull labels from the matching `labl` file, auto-detected
next to the step file (sibling `labl/` subdir, then flat layout). If no `labl` is
found, only the Energy Deposit mode renders.

```bash
# Default: 12s rotation, 30fps, 1440x1440, 100k points
jaxtpc-export-gif path/to/sim_step_0000.h5 --event 0

# Custom settings
jaxtpc-export-gif path/to/sim_step_0000.h5 \
    --event 0 --duration 6 --fps 30 \
    --size 1080 1080 --dpi 200 --max-points 80000 \
    --output my_event.gif

# Single volume, light background
jaxtpc-export-gif path/to/sim_step_0000.h5 -e 0 -v 0 --light

# MP4 output (requires ffmpeg)
jaxtpc-export-gif path/to/sim_step_0000.h5 -e 0 -o event.mp4
```

### Options

| Flag | Short | Default | Description |
|---|---|---|---|
| `step_file` | — | *(required)* | Path to a `*_step_*.h5` file |
| `--event` | `-e` | `0` | Event index |
| `--volume` | `-v` | all | Volume index (omit for all volumes) |
| `--output` | `-o` | `jaxtpc_3d.gif` | Output file (`.gif`, `.mp4`, `.webm`) |
| `--max-points` | — | `100000` | Max segments to render (subsampled, keeping high-dE) |
| `--fps` | — | `30` | Frames per second |
| `--duration` | — | `12.0` | Seconds per full rotation |
| `--rotations` | — | `1` | Number of full 360° rotations |
| `--dpi` | — | `200` | Render resolution (also affects point size) |
| `--size` | — | `1440 1440` | Frame size in pixels (`W H`) |
| `--emph-pow` | — | `5.0` | dE emphasis power (steepness) |
| `--emph-amt` | — | `0.75` | dE emphasis amount (`0`=uniform, `1`=full) |
| `--light` | — | off | Light background |
| `--labl` | — | auto | Override the `labl` path (auto-detected sibling by default) |

Output format is chosen from the `--output` extension. `.mp4`/`.webm` use
Matplotlib's `FFMpegWriter` and therefore need **ffmpeg** on the `PATH`; if it is
missing, export falls back to writing a `.gif`.

!!! tip "Render cost scales with points, not resolution"
    Frame time is driven by `--max-points`, not `--size`/`--dpi`. At the default
    100k points expect roughly a couple of seconds per frame, so a 12s / 30fps
    GIF (~360 frames) takes several minutes. Drop `--max-points` for quick
    previews. A progress bar with an ETA prints while rendering.

## Dependencies

- **Interactive viewer:** nothing extra on the Python side beyond the server; the
  browser loads `h5wasm` from a CDN.
- **GIF export:** `numpy`, `h5py`, `matplotlib`, `Pillow` (all standard). MP4
  output additionally needs `ffmpeg`. `hdf5plugin` is imported when available so
  step files written with the default `blosc-zstd` codec are readable.
